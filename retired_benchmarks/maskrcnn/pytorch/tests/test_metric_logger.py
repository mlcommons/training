# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import unittest

from maskrcnn_benchmark.utils.metric_logger import MetricLogger


class TestMetricLogger(unittest.TestCase):
    def test_update(self):
        meter = MetricLogger()
        for i in range(10):
            meter.update(metric=float(i))
        
        m = meter.meters["metric"]
        self.assertEqual(m.count, 10)
        self.assertEqual(m.total, 45)
        self.assertEqual(m.median, 4)
        self.assertEqual(m.avg, 4.5)

    def test_no_attr(self):
        meter = MetricLogger()
        _ = meter.meters
        _ = meter.delimiter
        def broken():
            _ = meter.not_existent
        self.assertRaises(AttributeError, broken)

if __name__ == "__main__":
    unittest.main()
