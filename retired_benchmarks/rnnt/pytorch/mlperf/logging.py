# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch

from mlperf_logging import mllog
from mlperf_logging.mllog import constants


mllogger = mllog.get_mllogger()


def configure_logger(benchmark):
    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{benchmark}.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False


def log_start(*args, **kwargs):
    _log(mllogger.start, *args, **kwargs)
def log_end(*args, **kwargs):
    _log(mllogger.end, *args, **kwargs)
def log_event(*args, **kwargs):
    _log(mllogger.event, *args, **kwargs)

def _log(logger, *args, **kwargs):
    """
    Wrapper for MLPerf compliance logging calls.
    All arguments but 'sync' and 'log_all_ranks' are passed to
    mlperf_logging.mllog.
    If 'sync' is set to True then the wrapper will synchronize all distributed
    workers. 'sync' should be set to True for all compliance tags that require
    accurate timing (RUN_START, RUN_STOP etc.)
    If 'log_all_ranks' is set to True then all distributed workers will print
    logging message, if set to False then only worker with rank=0 will print
    the message.
    """
    if 'stack_offset' not in kwargs:
        kwargs['stack_offset'] = 3
    if 'value' not in kwargs:
        kwargs['value'] = None

    if kwargs.pop('log_all_ranks', False):
        log = True
    else:
        log = (get_rank() == 0)

    if log:
        logger(*args, **kwargs)


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank

