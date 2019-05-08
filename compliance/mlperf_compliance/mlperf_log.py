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
"""Convenience function for logging compliance tags to stdout.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import logging
import json
import os
import re
import sys
import time
import uuid

from mlperf_compliance.tags import *
from mlperf_compliance import constant_sets
from mlperf_compliance import validations

# Default global variables can be set through setdefault()
BENCHMARK_ROOT_DIR = None
BENCHMARK_NAME = None
STACK_OFFSET = 1
LOG_LINE_EXTRA_PRINT = False
LOG_LINE_PREFIX = ""

# Constants
LOG_LINE_FORMAT = "{prefix}:::MLL {timestamp:.3f} {key}: {log_json}"
PATTERN = re.compile('[a-zA-Z0-9]+')

LOG_FILE = os.getenv("COMPLIANCE_FILE")
# create logger with 'spam_application'
LOGGER = logging.getLogger('mlperf_compliance')
LOGGER.setLevel(logging.DEBUG)

_STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)
_STREAM_HANDLER.setLevel(logging.INFO)
LOGGER.addHandler(_STREAM_HANDLER)

if LOG_FILE:
  _FILE_HANDLER = logging.FileHandler(LOG_FILE)
  _FILE_HANDLER.setLevel(logging.DEBUG)
  LOGGER.addHandler(_FILE_HANDLER)
else:
  _STREAM_HANDLER.setLevel(logging.DEBUG)

def setdefault(root_dir=None, benchmark=None, stack_offset=1,
               extra_print=False, prefix=""):
  """Set default attributes of mlperf logger"""
  global BENCHMARK_NAME
  global BENCHMARK_ROOT_DIR
  global STACK_OFFSET
  global LOG_LINE_EXTRA_PRINT
  global LOG_LINE_PREFIX

  if (benchmark is not None) and (
      benchmark not in constant_sets.BENCHMARK_NAME_SET):
    LOGGER.warning(
        "WARNING: \"{}\" is not in known benchmark names.".format(benchmark))

  BENCHMARK_NAME = benchmark
  BENCHMARK_ROOT_DIR = root_dir
  STACK_OFFSET = stack_offset
  LOG_LINE_EXTRA_PRINT = extra_print
  LOG_LINE_PREFIX = prefix

def get_caller(stack_index=2, root_dir=None):
  """Get caller's file and line number information.

  Arguments:
    stack_index: A stack_index of 2 will provide the caller of the
      function calling this function. Notice that stack_index of 2
      or more will fail if called from global scope.
    root_dir: The root dir to prefixed to the file name.

  Returns:
    Callsite info in a dictionary with these fields:
      "file": (string) file path
      "lineno": (int) line number
  """
  caller = inspect.getframeinfo(inspect.stack()[stack_index][0])

  # Trim the filenames for readability.
  filename = caller.filename
  if root_dir is not None:
    filename = re.sub("^" + root_dir + "/", "", filename)
  return {"file": filename, "lineno": caller.lineno}


def mlperf_print(
    key,
    value=None,
    metadata=None,
    deferred=False,
    benchmark=None,
    stack_offset=None,
    root_dir=None,
    extra_print=None,
    prefix=None):
  """Print out an MLPerf log line.

  Arguments:
    key: The MLPerf log key such as 'RUN_START' or 'EVAL_ACCURACY'. See the
      list of log keys in the spec.
    value: The value corresponding to the log key.
    metadata: The metadata corresponding to the log key.
    benchmark: The short code for the benchmark being run, see the MLPerf log spec.
    stack_offset: Increase the value to go deeper into the stack to find the
      callsite. For example, if this is being called by a wraper/helper you
      may want to set stack_offset=1 to use the callsite of the wraper/helper itself.
    deferred: The value is not presently known. In that case, a unique ID will
      be assigned as the value of this call and will be returned. The caller
      can then include said unique ID when the value is known later.
    root_dir: Directory prefix which will be trimmed when reporting calling
      file for compliance logging.
    extra_print: Print a blank line before logging to clear any text in the line.
    prefix: String with which to prefix the log message. Useful for
      differentiating raw lines if stitching will be required.

  Example output:
    :::MLL 1556733699.710 run_start: {"value": null, "metadata": {"lineno": 77, "file": "main.py"}}
  """

  # Get values from default attributes
  # the default attributes can be set by calling setdefault()
  benchmark = BENCHMARK_NAME if benchmark is None else benchmark
  stack_offset = STACK_OFFSET if stack_offset is None else stack_offset
  root_dir = BENCHMARK_ROOT_DIR if root_dir is None else root_dir
  extra_print = LOG_LINE_EXTRA_PRINT if extra_print is None else extra_print
  prefix = LOG_LINE_PREFIX if prefix is None else prefix

  return_value = None

  try:
    validations.validate(key, value, metadata, benchmark)
  except ValueError as e:
    LOGGER.warning("WARNING: Log validation: {}".format(e))

  log_meta = {}
  if metadata and isinstance(metadata, dict):
    log_meta.update(metadata)
  log_meta.update(get_caller(2 + stack_offset, root_dir=root_dir))
  log_json = json.dumps({"value": value, "metadata": log_meta})

  message = LOG_LINE_FORMAT.format(
      prefix=prefix, timestamp=time.time(), key=key, log_json=log_json)

  _do_print(key, message, extra_print)

  return return_value

def _do_print(key, message, extra_print=False):
  if extra_print:
    print() # There could be prior text on a line
  if key in constant_sets.STDOUT_KEY_SET:
    LOGGER.info(message)
  else:
    LOGGER.debug(message)

def _mlperf_print_v0_5(key, value=None, benchmark=None, stack_offset=0,
                  tag_set=None, deferred=False, root_dir=None,
                  extra_print=False, prefix=""):
  ''' Prints out an MLPerf Log Line.

  DEPRECATED: No longer applicable to v0.6

  key: The MLPerf log key such as 'CLOCK' or 'QUALITY'. See the list of log keys in the spec.
  value: The value which contains no newlines.
  benchmark: The short code for the benchmark being run, see the MLPerf log spec.
  stack_offset: Increase the value to go deeper into the stack to find the callsite. For example, if this
                is being called by a wraper/helper you may want to set stack_offset=1 to use the callsite
                of the wraper/helper itself.
  tag_set: The set of tags in which key must belong.
  deferred: The value is not presently known. In that case, a unique ID will
            be assigned as the value of this call and will be returned. The
            caller can then include said unique ID when the value is known
            later.
  root_dir: Directory prefix which will be trimmed when reporting calling file
            for compliance logging.
  extra_print: Print a blank line before logging to clear any text in the line.
  prefix: String with which to prefix the log message. Useful for
          differentiating raw lines if stitching will be required.

  Example output:
    :::MLP-1537375353 MINGO[17] (eval.py:42) QUALITY: 43.7
  '''

  return_value = None

  if (tag_set is None and not PATTERN.match(key)) or key not in tag_set:
    raise ValueError('Invalid key for MLPerf print: ' + str(key))

  if value is not None and deferred:
    raise ValueError("deferred is set to True, but a value was provided")

  if deferred:
    return_value = str(uuid.uuid4())
    value = "DEFERRED: {}".format(return_value)

  if value is None:
    tag = key
  else:
    str_json = json.dumps(value)
    tag = '{key}: {value}'.format(key=key, value=str_json)

  callsite = get_caller(2 + stack_offset, root_dir=root_dir)
  now = time.time()

  message = '{prefix}:::MLPv0.5.0 {benchmark} {secs:.9f} ({callsite}) {tag}'.format(
      prefix=prefix, secs=now, benchmark=benchmark, callsite=callsite, tag=tag)

  if extra_print:
    print() # There could be prior text on a line

  if tag in STDOUT_TAG_SET:
    LOGGER.info(message)
  else:
    LOGGER.debug(message)

  return return_value


GNMT_TAG_SET = set(GNMT_TAGS)
def gnmt_print(key, value=None, stack_offset=1, deferred=False, prefix=""):
  """DEPRECATED: No longer applicable to v0.6"""
  return _mlperf_print_v0_5(key=key, value=value, benchmark=GNMT,
                       stack_offset=stack_offset, tag_set=GNMT_TAG_SET,
                       deferred=deferred, root_dir=ROOT_DIR_GNMT)


MASKRCNN_TAG_SET = set(MASKRCNN_TAGS)
def maskrcnn_print(key, value=None, stack_offset=1, deferred=False,
    extra_print=True, prefix=""):
  """DEPRECATED: No longer applicable to v0.6"""
  return _mlperf_print_v0_5(key=key, value=value, benchmark=MASKRCNN,
                       stack_offset=stack_offset, tag_set=MASKRCNN_TAG_SET,
                       deferred=deferred, extra_print=extra_print,
                       root_dir=ROOT_DIR_MASKRCNN, prefix=prefix)


MINIGO_TAG_SET = set(MINIGO_TAGS)
def minigo_print(key, value=None, stack_offset=1, deferred=False, prefix=""):
  """DEPRECATED: No longer applicable to v0.6"""
  return _mlperf_print_v0_5(key=key, value=value, benchmark=MINIGO,
                       stack_offset=stack_offset, tag_set=MINIGO_TAG_SET,
                       deferred=deferred, root_dir=ROOT_DIR_MINIGO,
                       prefix=prefix)


NCF_TAG_SET = set(NCF_TAGS)
def ncf_print(key, value=None, stack_offset=1, deferred=False,
              extra_print=True, prefix=""):
  """DEPRECATED: No longer applicable to v0.6"""
  # Extra print is needed for the reference NCF because of tqdm.
  return _mlperf_print_v0_5(key=key, value=value, benchmark=NCF,
                       stack_offset=stack_offset, tag_set=NCF_TAG_SET,
                       deferred=deferred, extra_print=extra_print,
                       root_dir=ROOT_DIR_NCF, prefix=prefix)


RESNET_TAG_SET = set(RESNET_TAGS)
def resnet_print(key, value=None, stack_offset=1, deferred=False, prefix=""):
  """DEPRECATED: No longer applicable to v0.6"""
  return _mlperf_print_v0_5(key=key, value=value, benchmark=RESNET,
                       stack_offset=stack_offset, tag_set=RESNET_TAG_SET,
                       deferred=deferred, root_dir=ROOT_DIR_RESNET,
                       prefix=prefix)


SSD_TAG_SET = set(SSD_TAGS)
def ssd_print(key, value=None, stack_offset=1, deferred=False,
              extra_print=True, prefix=""):
  """DEPRECATED: No longer applicable to v0.6"""
  return _mlperf_print_v0_5(key=key, value=value, benchmark=SSD,
                       stack_offset=stack_offset, tag_set=SSD_TAG_SET,
                       deferred=deferred, extra_print=extra_print,
                       root_dir=ROOT_DIR_SSD, prefix=prefix)


TRANSFORMER_TAG_SET = set(TRANSFORMER_TAGS)
def transformer_print(key, value=None, stack_offset=1, deferred=False, prefix=""):
  """DEPRECATED: No longer applicable to v0.6"""
  return _mlperf_print_v0_5(key=key, value=value, benchmark=TRANSFORMER,
                       stack_offset=stack_offset, tag_set=TRANSFORMER_TAG_SET,
                       deferred=deferred, root_dir=ROOT_DIR_TRANSFORMER,
                       prefix=prefix)


if __name__ == '__main__':
  ncf_print(EVAL_ACCURACY, {'epoch': 7, 'accuracy': 43.7})
  ncf_print(INPUT_SIZE, 1024)
