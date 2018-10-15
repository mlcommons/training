# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Keys which only appear in ResNet.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


BOTTLENECK_BLOCK = "bottleneck_block"

# ==============================================================================
# == Topology ==================================================================
# ==============================================================================

MODEL_HP_INITIAL_MAX_POOL = "model_hp_initial_max_pool"
MODEL_HP_BEGIN_BLOCK = "model_hp_begin_block"
MODEL_HP_END_BLOCK = "model_hp_end_block"
MODEL_HP_BLOCK_TYPE = "model_hp_block_type"
MODEL_HP_PROJECTION_SHORTCUT = "model_hp_projection_shortcut"
MODEL_HP_SHORTCUT_ADD = "model_hp_shorcut_add"

MODEL_HP_RESNET_TOPOLOGY = "model_hp_resnet_topology"
