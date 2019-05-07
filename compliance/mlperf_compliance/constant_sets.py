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

from mlperf_compliance import constants

BENCHMARK_NAME_SET = set([
  constants.RESNET,
  constants.SSD,
  constants.MASKRCNN,
  constants.GNMT,
  constants.TRANSFORMER,
  constants.MINIGO
])

COMMON_KEY_SET = set([
  constants.SUBMISSION_ORG,
  constants.SUBMISSION_PLATFORM,
  constants.SUBMISSION_STATUS,
  constants.SUBMISSION_BENCHMARK,
  constants.SUBMISSION_POC_NAME,
  constants.SUBMISSION_POC_EMAIL,
  constants.SUBMISSION_ENTRY,

  constants.BLOCK_START,
  constants.BLOCK_STOP,
  constants.EPOCH_START,
  constants.EPOCH_STOP,
  constants.EVAL_START,
  constants.EVAL_STOP,
  constants.INIT_START,
  constants.INIT_STOP,
  constants.RUN_START,
  constants.RUN_STOP,
  constants.CACHE_CLEAR,
  constants.EVAL_ACCURACY
])

STDOUT_KEY_SET = set([
  constants.BLOCK_START,
  constants.BLOCK_STOP,
  constants.EPOCH_START,
  constants.EPOCH_STOP,
  constants.EVAL_START,
  constants.EVAL_STOP,
  constants.INIT_START,
  constants.INIT_STOP,
  constants.RUN_START,
  constants.RUN_STOP,
  constants.CACHE_CLEAR,
  constants.EVAL_ACCURACY
])

RESNET_KEY_SET = set.union(COMMON_KEY_SET, set([
  constants.GLOBAL_BATCH_SIZE,
  constants.OPT_NAME,
  constants.OPT_BASE_LR,
  constants.OPT_LR_WARMUP_EPOCHS,
  constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
  constants.LARS_OPT_END_LR,
  constants.LARS_OPT_LR_DECAY_STEPS,
  constants.LARS_OPT_LR_DECAY_POLY_POWER,
  constants.LARS_EPSILON,
  constants.MODEL_BN_SPAN
]))

SSD_KEY_SET = set.union(COMMON_KEY_SET, set([
  constants.GLOBAL_BATCH_SIZE,
  constants.OPT_BASE_LR,
  constants.OPT_WEIGHT_DECAY,
  constants.OPT_LR_WARMUP_STEPS,
  constants.OPT_LR_WARMUP_FACTOR,
  constants.MAX_SAMPLES,
  constants.MODEL_BN_SPAN
]))

MASKRCNN_KEY_SET = set.union(COMMON_KEY_SET, set([
  constants.GLOBAL_BATCH_SIZE,
  constants.OPT_BASE_LR,
  constants.OPT_LR_WARMUP_STEPS,
  constants.OPT_LR_WARMUP_FACTOR,
  constants.NUM_IMAGE_CANDIDATES
]))

GNMT_KEY_SET = set.union(COMMON_KEY_SET, set([
  constants.GLOBAL_BATCH_SIZE,
  constants.OPT_LR_ALT_DECAY_FUNC,
  constants.OPT_BASE_LR,
  constants.OPT_LR_DECAY_INTERVAL,
  constants.OPT_LR_DECAY_FACTOR,
  constants.OPT_LR_DECAY_STEPS,
  constants.OPT_LR_REMAIN_STEPS,
  constants.OPT_LR_ALT_WARMUP_FUNC,
  constants.OPT_LR_WARMUP_STEPS,
  constants.MAX_SEQUENCE_LENGTH
]))

TRANSFORMER_KEY_SET = set.union(COMMON_KEY_SET, set([
  constants.GLOBAL_BATCH_SIZE,
  constants.OPT_NAME,
  constants.OPT_BASE_LR,
  constants.OPT_LR_WARMUP_STEPS,
  constants.MAX_SEQUENCE_LENGTH,
  constants.OPT_ADAM_BETA_1,
  constants.OPT_ADAM_BETA_2,
  constants.OPT_ADAM_EPSILON
]))

MINIGO_KEY_SET = set.union(COMMON_KEY_SET, set([
  constants.GLOBAL_BATCH_SIZE,
  constants.OPT_BASE_LR,
  constants.OPT_LR_DECAY_BOUNDARY_STEPS
]))
