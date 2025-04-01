# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import time

import torch

from megatron import print_rank_0, initialize_megatron, get_args, get_timers, mpu
from megatron.initialize import _compile_dependencies
from megatron.model import GPTModel, ModelType
from megatron.training import print_datetime, setup_model_and_optimizer


def initialize(model_provider,
               model_type,
               extra_args_provider=None,
               args_defaults={},
               compile_dependencies=True):

    init_time = time.time()
    # Initalize and get arguments, timers, and Tensorboard writer.
    finish_mpu_init = initialize_megatron(extra_args_provider=extra_args_provider,
                                          args_defaults={**args_defaults, 'lazy_mpu_init': True})
    finish_mpu_init()
    if compile_dependencies:
        _compile_dependencies()

    print_rank_0(f'time to initialize megatron (seconds): {time.time() - init_time:.3f}')

    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)
    timers('model-and-optimizer-setup').stop()

    # Print setup timing.
    print_rank_0('done with setup ...')

    return model, optimizer, opt_param_scheduler


def init_optimizer_state(opt):
    for group in opt.param_groups:
        for p in group['params']:
            if p is not None:
                p.grad = torch.rand_like(p.data)

    args = get_args()
    timers = get_timers()

    opt.step(args, timers)


def gpt_model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model
