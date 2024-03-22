import os
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
from mlperf_logging.mllog.mllog import MLLogger

def get_mlperf_logger(path, filename='mlperf_gnn.log'):
    mllog.config(filename=os.path.join(path, filename))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    return mllogger

def submission_info(mllogger: MLLogger, benchmark_name: str, submitter_name: str):
    """Logs required for a valid MLPerf submission."""
    mllogger.event(
        key=constants.SUBMISSION_BENCHMARK,
        value=benchmark_name,
    )
    mllogger.event(
        key=constants.SUBMISSION_ORG,
        value=submitter_name,
    )
    mllogger.event(
        key=constants.SUBMISSION_DIVISION,
        value=constants.CLOSED,
    )
    mllogger.event(
        key=constants.SUBMISSION_STATUS,
        value=constants.ONPREM,
    )
    mllogger.event(
        key=constants.SUBMISSION_PLATFORM,
        value=submitter_name,
    )
