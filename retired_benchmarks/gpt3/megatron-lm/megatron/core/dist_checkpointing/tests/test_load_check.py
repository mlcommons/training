# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from datetime import datetime

import torch

from megatron import print_rank_0, get_args, get_timers, initialize_megatron
from megatron.core import dist_checkpointing
from megatron.checkpointing import load_checkpoint
from megatron.core.dist_checkpointing import diff
from megatron.core.dist_checkpointing import gpt_model_provider, \
    init_optimizer_state
from megatron.core.dist_checkpointing.tests.test_save import get_checkpoint_dir
from megatron.model import ModelType
from megatron.mpu import model_parallel_cuda_manual_seed
from megatron.training import setup_model_and_optimizer


def test_load_correctness(model_provider, expected_seed=1, mismatch_seed=2, print_timers=True):

    timers = get_timers()
    args = get_args()

    # Model, optimizer, and learning rate.

    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model_parallel_cuda_manual_seed(expected_seed)
    exp_model, exp_optimizer, exp_opt_param_scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
    timers('model-and-optimizer-setup').stop()

    model_parallel_cuda_manual_seed(mismatch_seed)
    mis_model, mis_optimizer, mis_opt_param_scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

    # Print setup timing.
    print_rank_0('done with setup ...')

    init_optimizer_state(exp_optimizer)
    init_optimizer_state(mis_optimizer)


    checkpointing_config = dist_checkpointing.config.config_from_args(args, 'save')
    assert checkpointing_config is not None

    checkpoint_dir = get_checkpoint_dir(checkpointing_config, args, model_provider.__name__, rm=False)
    args.load = checkpoint_dir
    timers_fpath = f'{checkpoint_dir}-{datetime.now().isoformat()}.timers'

    only_left, only_right, mismatch = diff(exp_model[0].state_dict(), mis_model[0].state_dict())
    assert not only_right and not only_right
    # The only mismatch is tensors difference
    assert len(mismatch) > 0
    print_rank_0(mismatch)

    print_rank_0('Checking optimizer diffs before load')
    only_left, only_right, mismatch = diff(exp_optimizer.state_dict()['fp32_from_fp16_params'], mis_optimizer.state_dict()['fp32_from_fp16_params'])
    assert len(only_left) == len(only_right) == 0
    assert len(mismatch) > 0


    torch.distributed.barrier()
    print_rank_0('start loading ...')
    timers('load-checkpoint', log_level=0).start(barrier=True)
    iteration = load_checkpoint(mis_model, mis_optimizer, None)
    timers('load-checkpoint').stop()
    assert iteration == 10

    print_rank_0('Checking diffs after load')
    only_left, only_right, mismatch = diff(mis_model[0].state_dict_for_save_checkpoint(), exp_model[0].state_dict_for_save_checkpoint())
    assert not only_right and not only_right
    # It's not true for different expected model initialization
    # assert len(mismatch) == 0, f'[{torch.distributed.get_rank()}] mismatch len == {len(mismatch)}, {mismatch}'

    print_rank_0('Checking optimizer diffs after load')
    only_left, only_right, mismatch = diff(exp_optimizer.state_dict()['fp32_from_fp16_params'], mis_optimizer.state_dict()['fp32_from_fp16_params'])
    assert len(only_left) == len(only_right) == 0
    # It's not true for different expected model initialization
    # assert len(mismatch) == 0, mismatch

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
    test_load_correctness(gpt_model_provider, 1, 2, print_timers=True)
