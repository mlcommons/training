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
from mlperf_compliance import constant_sets

benchmark_key_sets = {
  constants.RESNET: constant_sets.RESNET_KEY_SET,
  constants.SSD: constant_sets.SSD_KEY_SET,
  constants.MASKRCNN: constant_sets.MASKRCNN_KEY_SET,
  constants.GNMT: constant_sets.GNMT_KEY_SET,
  constants.TRANSFORMER: constant_sets.TRANSFORMER_KEY_SET,
  constants.MINIGO: constant_sets.MINIGO_KEY_SET
}

def validate(key, value, metadata, benchmark=None):
  key_set = set()
  if benchmark is not None and benchmark in benchmark_key_sets:
    key_set = benchmark_key_sets[benchmark]
    if key not in key_set:
      raise ValueError(
          "Key \"{}\" is not in known {} keys.".format(key, benchmark))
  return True
