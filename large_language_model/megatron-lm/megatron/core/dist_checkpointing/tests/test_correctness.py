# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

import megatron.core.dist_checkpointing.load_save
from megatron import print_rank_0, get_args, get_timers, initialize_megatron
from megatron.core import dist_checkpointing
from megatron.checkpointing import generate_model_optim_state_dicts
from megatron.core.dist_checkpointing import diff
from megatron.core.dist_checkpointing import gpt_model_provider, \
    init_optimizer_state
from megatron.model import ModelType
from megatron.training import setup_model_and_optimizer


def get_checkpoint_dir(checkpointing_config, args, model_name, rm=True):
    dirname = Path(args.save)
    ckpt_dir = dirname / f'{model_name}-{checkpointing_config.backend}-{checkpointing_config.strategy}'
    if rm and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir


def test_save_load_correctness(model_provider, init_first_optim=True, init_second_optim=True, print_timers=True):

    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
    timers('model-and-optimizer-setup').stop()

    # Print setup timing.
    print_rank_0('done with setup ...')

    if init_first_optim:
        init_optimizer_state(optimizer)

    first_optim_state_dict = deepcopy(optimizer.state_dict())

    args = get_args()

    checkpointing_config = dist_checkpointing.config.config_from_args(args, 'save')
    assert checkpointing_config is not None

    checkpoint_dir = get_checkpoint_dir(checkpointing_config, args, model_provider.__name__)
    timers_fpath = f'{checkpoint_dir}-{datetime.now().isoformat()}.timers'

    # Torch
    timers('generate-state-dict', log_level=0).start(barrier=True)
    model_state_dict, optim_state_dict = generate_model_optim_state_dicts(model, optimizer, opt_param_scheduler, True)
    state_dict = {**model_state_dict, **optim_state_dict}
    timers('generate-state-dict').stop()

    timers('save-dist-checkpointing', log_level=0).start(barrier=True)
    megatron.core.dist_checkpointing.load_save.save(state_dict, checkpoint_dir, checkpointing_config)
    timers('save-dist-checkpointing').stop()

    del checkpointing_config

    print_rank_0('done saving ...')
    # timers_path = get_timers_fname(checkpointing_config, args, 'gpt')

    new_model, new_optimizer, new_opt_param_scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

    only_left, only_right, mismatch = diff(model[0].state_dict(), new_model[0].state_dict())
    assert not only_right and not only_right
    # The only mismatch is tensors difference
    assert len(mismatch) > 0
    print_rank_0(mismatch)

    print_rank_0('Checking optimizer diffs before load')
    if init_second_optim:
        init_optimizer_state(new_optimizer)
    only_left, only_right, mismatch = diff(first_optim_state_dict['optimizer']['state'], new_optimizer.state_dict()['optimizer']['state'])
    if init_first_optim > init_second_optim:
        assert len(only_left) > 0
        assert len(only_right) == 0, only_right
    elif init_first_optim < init_second_optim:
        assert len(only_left) == 0, only_left
        assert len(only_right) > 0
    else:
        assert len(only_left) == 0, only_left
        assert len(only_right) == 0, only_right

    only_left, only_right, mismatch = diff(first_optim_state_dict['fp32_from_fp16_params'], new_optimizer.state_dict()['fp32_from_fp16_params'])
    assert len(only_left) == len(only_right) == 0
    assert len(mismatch) > 0


    torch.distributed.barrier()
    print_rank_0('start loading ...')
    checkpointing_config = dist_checkpointing.config.config_from_args(args, 'load')
    new_model_state_dict, new_optim_state_dict = generate_model_optim_state_dicts(new_model, new_optimizer, new_opt_param_scheduler, True)
    new_sharded_state_dict = {**new_model_state_dict, **new_optim_state_dict}

    timers('load-dist-checkpointing', log_level=0).start(barrier=True)
    new_state_dict = megatron.core.dist_checkpointing.load_save.load(new_sharded_state_dict, checkpoint_dir, checkpointing_config)
    timers('load-dist-checkpointing').stop()

    timers('load-state-dict', log_level=0).start(barrier=True)
    new_model[0].load_state_dict(new_state_dict['model'])
    timers('load-state-dict').stop()

    only_left, only_right, mismatch = diff(model[0].state_dict_for_save_checkpoint(), new_model[0].state_dict_for_save_checkpoint())
    assert not only_right and not only_right
    assert len(mismatch) == 0, f'[{torch.distributed.get_rank()}] mismatch len == {len(mismatch)}, {mismatch}'

    timers('load-optim-state-dict', log_level=0).start(barrier=True)
    new_optimizer.load_state_dict(new_state_dict['optimizer'])
    timers('load-optim-state-dict').stop()

    print_rank_0('Checking optimizer diffs after load')
    only_left, only_right, mismatch = diff(first_optim_state_dict['fp32_from_fp16_params'], new_optimizer.state_dict()['fp32_from_fp16_params'])
    assert len(only_left) == len(only_right) == 0
    assert len(mismatch) == 0, mismatch
    if init_first_optim and init_second_optim:
        only_left, only_right, mismatch = diff(first_optim_state_dict['optimizer']['state'], new_optimizer.state_dict()['optimizer']['state'])
        assert len(only_left) == len(only_right) == 0
        assert len(mismatch) == 0, mismatch

    torch.distributed.barrier()

    print_rank_0('Test passed')

    if print_timers:
        timers.log(timers._timers.keys(), rank=0)

    # if torch.distributed.get_rank() == 0:
    #     f = open(timers_fpath, 'w')
    #     f_ctx = f
    #     ctx = redirect_stdout(f)
    # else:
    #     f_ctx = nullcontext()
    #     ctx = nullcontext()
    # with f_ctx, ctx:
    #     timers.log(timers._timers.keys(), rank=0)


if __name__ == "__main__":
    finish_mpu_init = initialize_megatron(extra_args_provider=dist_checkpointing.config.add_argparse_args,
                                          args_defaults={'tokenizer_type': 'GPT2BPETokenizer', 'lazy_mpu_init': False})
    for init_first_optim in [True, False]:
        for init_second_optim in [True, False]:
            torch.distributed.barrier()
            print('#' * 100)
            print('Start test', init_first_optim, init_second_optim)
            test_save_load_correctness(gpt_model_provider, init_first_optim,
                                       init_second_optim, print_timers=False)
