import os
from functools import wraps

import utils
from mlperf_logging import mllog


class SSDLogger:
    def __init__(self, filename=None, default_stack_offset=2):
        self.mllogger = mllog.get_mllogger()
        mllog.config(default_stack_offset=default_stack_offset,
                     filename=(filename or os.getenv("COMPLIANCE_FILE") or "mlperf_compliance.log"),
                     root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))))

    @property
    def rank(self):
        return utils.get_rank()

    def event(self, sync=False, log_rank=None, *args, **kwargs):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            utils.barrier()
        if log_rank:
            self.mllogger.event(*args, **kwargs)

    def start(self, sync=False, log_rank=None, *args, **kwargs):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            utils.barrier()
        if log_rank:
            self.mllogger.start(*args, **kwargs)

    def end(self, sync=False, log_rank=None, *args, **kwargs):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            utils.barrier()
        if log_rank:
            self.mllogger.end(*args, **kwargs)


mllogger = SSDLogger()
