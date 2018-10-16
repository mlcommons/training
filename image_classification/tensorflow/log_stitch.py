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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import json
import sys

# ID format of uuid.uuid4()
_UUID_PATTERN = "[0-9a-z]{8}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{12}"

# Prefix of all mlperf logging lines
_COMMON_PATTERN = "^:::MLPv0\.5\.[0-9]+"

# Currently this script is only written to stitch resnet lines together. If it
# is of general utility, this can be extended to a pattern.
_BENCHMARK_NAME = "resnet"

PREFIX = re.compile(_COMMON_PATTERN)
DEFERRED_DECLARATION = re.compile(
    "({} {} )([0-9\.]+)( .+)\"DEFERRED: ({})\"$".format(
        _COMMON_PATTERN, _BENCHMARK_NAME, _UUID_PATTERN))
DEFERRED_PREFIX = re.compile("{} \[({})\].+".format(_COMMON_PATTERN, _UUID_PATTERN))

# Info is logged as [timestamp][value]
DEFERRED_INFO_PATTERN = re.compile("^\[([0-9\.]+)\]\[([0-9\.e\-]+)\]")


def main():
  raw_log_lines = set()
  for line in sys.stdin:
    line = line.strip()
    # Do not repeat lines that are written to multiple sources. The timestamp
    # guarantees lines are not improperly dropped.
    if not PREFIX.match(line) or line in raw_log_lines:
      continue
    raw_log_lines.add(line)

  raw_log_lines = sorted(raw_log_lines)

  deferred_declarations = {}
  deferred_evaluations = []
  immediate_log_lines = []
  for line in raw_log_lines:
    if DEFERRED_PREFIX.match(line):
      deferred_evaluations.append(line)
      continue

    match = DEFERRED_DECLARATION.match(line)
    if match:
      _, timestamp, _, key = match.groups()
      assert key not in deferred_declarations
      deferred_declarations[key] = (line, timestamp)
      continue

    immediate_log_lines.append(line)

  for line in deferred_evaluations:
    match = DEFERRED_PREFIX.match(line)
    assert match
    key = match.groups()[0]

    # Strip off the id used for stitching
    logged_eval_info = re.sub("{} \[{}\]".format(_COMMON_PATTERN, key), "", line)
    # print(logged_eval_info, DEFERRED_INFO_PATTERN.match(logged_eval_info))
    eval_timestamp, eval_value = DEFERRED_INFO_PATTERN.match(logged_eval_info).groups()
    eval_value = float(eval_value)


    declaration_line, declaration_timestamp = deferred_declarations[key]

    # We replace the graph construction time with the eval call time, and add
    # in the reported value
    stitched_line = DEFERRED_DECLARATION.sub(
        r'\g<1>{}\g<3>'.format(eval_timestamp), declaration_line) + \
                    json.dumps({"value": eval_value, "deferred": True})

    # The stitched line can now be treated as an immediately logged line
    immediate_log_lines.append(stitched_line)

  immediate_log_lines.sort()
  for i in immediate_log_lines:
    print(i)
  sys.stdout.flush()


if __name__ == "__main__":
  main()
