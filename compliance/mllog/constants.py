# Copyright 2019 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Master list of constants in MLPerf log
"""

# NOTE: Keep string values in alphabetical order under each section.

# Constant values - log settings
DEFAULT_LOGGER_NAME = "mllog_default"
DEFAULT_NAMESPACE = ""

# Constant values - log event type
INTERVAL_END = "INTERVAL_END"
INTERVAL_START = "INTERVAL_START"
POINT_IN_TIME = "POINT_IN_TIME"

# Constant values - benchmark name
GNMT = "gnmt"
MASKRCNN = "maskrcnn"
MINIGO = "minigo"
NCF = "ncf"
RESNET = "resnet"
SSD = "ssd"
TRANSFORMER = "transformer"

# Constant values - model info
ADAM = "adam"
LARS = "lars"
LAZY_ADAM = "lazy_adam"
SGD = "sgd"

# Constant values - metadata info
ABORTED = "aborted"
SUCCESS = "success"

# Log keys - submission info
SUBMISSION_BENCHMARK = "submission_benchmark"
SUBMISSION_DIVISION = "submission_division"
SUBMISSION_ENTRY = "submission_entry"
SUBMISSION_ORG = "submission_org"
SUBMISSION_PLATFORM = "submission_platform"
SUBMISSION_POC_NAME = "submission_poc_name"
SUBMISSION_POC_EMAIL = "submission_poc_email"
SUBMISSION_STATUS = "submission_status"

# Log keys - timing info
BLOCK_START = "block_start"
BLOCK_STOP = "block_stop"
EPOCH_START = "epoch_start"
EPOCH_STOP = "epoch_stop"
EVAL_START = "eval_start"
EVAL_STOP = "eval_stop"
INIT_START = "init_start"
INIT_STOP = "init_stop"
RUN_START = "run_start"
RUN_STOP = "run_stop"

# Log keys - common run info
CACHE_CLEAR = "cache_clear"
EVAL_ACCURACY = "eval_accuracy"

# Log kyes - model hyperparameters
GLOBAL_BATCH_SIZE = "global_batch_size"
LARS_EPSILON = "lars_epsilon"
LARS_OPT_END_LR = "lars_opt_end_learning_rate"
LARS_OPT_LR_DECAY_POLY_POWER = "lars_opt_learning_rate_decay_poly_power"
LARS_OPT_LR_DECAY_STEPS = "lars_opt_learning_rate_decay_steps"
LARS_OPT_WEIGHT_DECAY = "lars_opt_weight_decay"
MAX_SAMPLES = "max_samples"
MAX_SEQUENCE_LENGTH = "max_sequence_length"
MODEL_BN_SPAN = "model_bn_span"
NUM_IMAGE_CANDIDATES = "num_image_candidates"
OPT_ADAM_BETA_1 = "opt_adam_beta_1"
OPT_ADAM_BETA_2 = "opt_adam_beta_2"
OPT_ADAM_EPSILON = "opt_adam_epsilon"
OPT_NAME = "opt_name"
OPT_BASE_LR = "opt_base_learning_rate"
OPT_LR_ALT_DECAY_FUNC = "opt_learning_rate_alt_decay_func"
OPT_LR_ALT_WARMUP_FUNC = "opt_learning_rate_alt_warmup_func"
OPT_LR_DECAY_BOUNDARY_EPOCHS = "opt_learning_rate_decay_boundary_epochs"
OPT_LR_DECAY_BOUNDARY_STEPS = "opt_learning_rate_decay_boundary_steps"
OPT_LR_DECAY_FACTOR = "opt_learning_rate_decay_factor"
OPT_LR_DECAY_INTERVAL = "opt_learning_rate_decay_interval"
OPT_LR_DECAY_STEPS = "opt_learning_rate_decay_steps"
OPT_LR_REMAIN_STEPS = "opt_learning_rate_remain_steps"
OPT_LR_WARMUP_EPOCHS = "opt_learning_rate_warmup_epochs"
OPT_LR_WARMUP_FACTOR = "opt_learning_rate_warmup_factor"
OPT_LR_WARMUP_STEPS = "opt_learning_rate_warmup_steps"
OPT_WEIGHT_DECAY = "opt_weight_decay"

# Log metadata keys
EPOCH_COUNT = "epoch_count"
EPOCH_NUM = "epoch_num"
FIRST_EPOCH_NUM = "first_epoch_num"
STATUS = "status"
