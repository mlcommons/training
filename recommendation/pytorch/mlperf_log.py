from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import re
import time
import json


# ==============================================================================
# == All Models ================================================================
# ==============================================================================
RUN_START = "run_start"
RUN_STOP = "run_stop"
RUN_FINAL = "run_final"

INPUT_SIZE = "input_size"
INPUT_BATCH_SIZE = "input_batch"
INPUT_ORDER = "input_order"

TRAIN_LOOP = "train_loop"
TRAIN_EPOCH = "train_epoch"
TRAIN_LEARN_RATE = "train_learn_rate"
TRAIN_CHECKPOINT = "train_checkpoint"

EVAL_START = "eval_start"
EVAL_SIZE = "eval_size"
EVAL_TARGET = "eval_target"
EVAL_ACCURACY = "eval_accuracy"
EVAL_STOP = "eval_stop"

OPT_NAME = "opt_name"

MODEL_HP_LOSS_FN = "model_hp_loss_fn"


# ==============================================================================
# == Common Values =============================================================
# ==============================================================================
BCE = "binary_cross_entropy"
CCE = "categorical_cross_entropy"


# NCF Recommendation
NCF = "ncf"

PREPROC_HP_MIN_RATINGS = "preproc_hp_min_ratings"
PREPROC_HP_NUM_NEG = "preproc_hp_num_neg"
PREPROC_HP_NUM_EVAL = "preproc_hp_num_eval"
PREPROC_HP_SAMPLE_EVAL_REPLACEMENT = "preproc_hp_sample_eval_replacement"
PREPROC_STEP_TRAIN_NEG_GEN = "preproc_step_train_neg_gen"
PREPROC_STEP_EVAL_NEG_GEN = "preproc_step_eval_neg_gen"

INPUT_HP_SAMPLE_TRAIN_REPLACEMENT = "input_hp_sample_train_replacement"

EVAL_HP_NUM_USERS = "eval_hp_num_users"
EVAL_HP_NUM_NEG = "eval_hp_num_neg"

OPT_HP_BETA1 = "opt_hp_beta1"
OPT_HP_BETA2 = "opt_hp_beta2"
OPT_HP_EPSILON = "opt_hp_epsilon"

MODEL_HP_MF_DIM = "model_hp_mf_dim"
MODEL_HP_MLP_LAYER_SIZES = "model_hp_mlp_layer_sizes"



def get_caller(stack_index=2):
  ''' Returns file.py:lineno of your caller. A stack_index of 2 will provide
      the caller of the function calling this function. Notice that stack_index
      of 2 or more will fail if called from global scope. '''
  caller = inspect.getframeinfo(inspect.stack()[stack_index][0])
  return "%s:%d" % (caller.filename, caller.lineno)


def mlperf_print(key, value=None, benchmark=None, stack_offset=0):
  ''' Prints out an MLPerf Log Line.

  key: The MLPerf log key such as 'CLOCK' or 'QUALITY'. See the list of log keys in the spec.
  value: The value which contains no newlines.
  benchmark: The short code for the benchmark being run, see the MLPerf log spec.
  stack_offset: Increase the value to go deeper into the stack to find the callsite. For example, if this
                is being called by a wraper/helper you may want to set stack_offset=1 to use the callsite
                of the wraper/helper itslef.

  Example output:
    :::MLP-1537375353 MINGO[17] (eval.py:42) QUALITY: 43.7
  '''
  if not re.match('[a-zA-Z0-9]+', key):
    raise ValueError('Invalid key for MLPerf print: ' + str(key))

  if value is None:
    tag = key
  else:
    str_json = json.dumps(value)
    tag = '{key}: {value}'.format(key=key, value=str_json)

  callsite = get_caller(2 + stack_offset)
  now = int(time.time())

  message = ':::MLPv0.5.0 {benchmark} {secs} ({callsite}) {tag}'.format(
      secs=now, benchmark=benchmark, callsite=callsite, tag=tag)

  # And log to tensorflow too
  print()  # There could be prior text on a line
  print(message)


ncf_print = functools.partial(mlperf_print, benchmark=NCF)


if __name__ == '__main__':
  mlperf_print('eval_accuracy', {'epoch': 7, 'accuracy': 43.7})
  mlperf_print('train_batch_size', 1024)
