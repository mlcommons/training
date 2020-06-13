# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a reinforcement learning loop to train a Go playing model."""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
import itertools
import logging
import os
import re
import tensorflow as tf
import time
from ml_perf.utils import *

from mlperf_logging import mllog

from absl import app, flags

flags.DEFINE_integer('iterations', 100, 'Number of iterations of the RL loop.')

flags.DEFINE_string('flags_dir', None,
                    'Directory in which to find the flag files for each stage '
                    'of the RL loop. The directory must contain the following '
                    'files: bootstrap.flags, selfplay.flags, eval.flags, '
                    'train.flags.')

flags.DEFINE_integer('window_size', 5,
                     'Maximum number of recent selfplay rounds to train on.')

flags.DEFINE_float('train_filter', 0.3,
                   'Fraction of selfplay games to pass to training.')

flags.DEFINE_integer('examples_per_generation', 131072,
                     'Number of examples use from each generation in the '
                     'training window.')

flags.DEFINE_boolean('validate', False, 'Run validation on holdout games')

flags.DEFINE_integer('min_games_per_iteration', 4096,
                     'Minimum number of games to play for each training '
                     'iteration.')

flags.DEFINE_integer('num_read_threads', 8,
                     'Number of threads to read examples on. Using more '
                     'read threads may speed up reading the examples as '
                     'more can be decompressed in parallel. This flag has '
                     'no effect on the output data.')

flags.DEFINE_integer('num_write_threads', 8,
                     'Number of threads to write examples on. Each thread '
                     'will write a separate .tfrecord.zz file to train on. '
                     'Using more threads may reduce the time take to generate '
                     'the training chunks as more threads are used to '
                     'compress the data. Using too many threads however could '
                     'slow down training time if each shard gets much smaller '
                     'than around 100MB.')

flags.DEFINE_integer(
    'mlperf_num_games', -1,
    'Total number of games to self play. 0 implies self '
    'play forever.')
flags.DEFINE_integer(
    'mlperf_num_readouts', -1,
    'Number of readouts to make during tree search for '
    'each move.')
flags.DEFINE_float(
    'mlperf_value_init_penalty', -1.0,
    'New children value initialization penalty. '
    'Child value = parent\'s value - penalty * color, '
    'clamped to [-1, 1].  Penalty should be in [0.0, 2.0]. '
    '0 is init-to-parent, 2.0 is init-to-loss [default]. '
    'This behaves similiarly to Leela\'s FPU '
    '"First Play Urgency".')
flags.DEFINE_float('mlperf_holdout_pct', -1.0,
                   'Fraction of games to hold out for validation.')
flags.DEFINE_float('mlperf_disable_resign_pct', -1.0,
                   'Fraction of games to disable resignation for.')
flags.DEFINE_multi_float(
    'mlperf_resign_threshold', [0.0, 0.0],
    'Each game\'s resign threshold is picked randomly '
    'from the range '
    '[min_resign_threshold, max_resign_threshold)')
flags.DEFINE_integer(
    'mlperf_parallel_games', -1,
    'Number of games to play concurrently on each selfplay '
    'thread. Inferences from a thread\'s concurrent games are '
    'batched up and evaluated together. Increasing '
    'concurrent_games_per_thread can help improve GPU or '
    'TPU utilization, especially for small models.')
flags.DEFINE_integer('mlperf_virtual_losses', -1,
                     'Number of virtual losses when running tree search')
flags.DEFINE_float(
    'mlperf_gating_win_rate', -1.0,
    'Win pct against the target model to define a converged '
    'model.')
flags.DEFINE_integer(
    'mlperf_eval_games', -1, 'Number of games to play against the target to '
    'when determining the win pct.')

flags.DEFINE_string('golden_chunk_dir', None, 'Training example directory.')
flags.DEFINE_string('holdout_dir', None, 'Holdout example directory.')
flags.DEFINE_string('model_dir', None, 'Model directory.')
flags.DEFINE_string('selfplay_dir', None, 'Selfplay example directory.')
flags.DEFINE_string('work_dir', None, 'Training work directory.')

flags.DEFINE_string('tpu_name', None, 'Name of the TPU to train on.')

# Flags defined in other files.
flags.declare_key_flag('train_batch_size')
flags.declare_key_flag('l2_strength')
flags.declare_key_flag('filter_amount')
flags.declare_key_flag('lr_boundaries')
flags.declare_key_flag('lr_rates')

FLAGS = flags.FLAGS


# Training loop state.
class State:
    def __init__(self, model_num):
        self.start_time = time.time()
        self.start_iter_num = model_num
        self.iter_num = model_num

    def _model_name(self, it):
        return '%06d' % it

    @property
    def selfplay_model_name(self):
        return self._model_name(self.iter_num - 1)

    @property
    def selfplay_model_path(self):
        return '{}.pb'.format(
            os.path.join(FLAGS.model_dir, self.selfplay_model_name))

    @property
    def train_model_name(self):
        return self._model_name(self.iter_num)

    @property
    def train_model_path(self):
        return '{}.pb'.format(
            os.path.join(FLAGS.model_dir, self.train_model_name))


def wait_for_training_examples(state, num_games):
    """Wait for training examples to be generated by the latest model.

    Args:
        state: the RL loop State instance.
        num_games: number of games to wait for.
    """

    model_dir = os.path.join(FLAGS.selfplay_dir, state.selfplay_model_name)
    pattern = os.path.join(model_dir, '*', '*', '*.tfrecord.zz')
    for i in itertools.count():
        try:
            paths = sorted(tf.gfile.Glob(pattern))
        except tf.errors.OpError:
            paths = []
        if len(paths) >= num_games:
            break
        if i % 30 == 0:
            logging.info('Waiting for %d games in %s (found %d)',
                         num_games, model_dir, len(paths))
        time.sleep(1)


def list_selfplay_dirs(base_dir):
    """Returns a sorted list of selfplay data directories.

    Training examples are written out to the following directory hierarchy:
      base_dir/device_id/model_name/timestamp/

    Args:
      base_dir: either selfplay_dir or holdout_dir.

    Returns:
      A list of model directories sorted so the most recent directory is first.
    """

    model_dirs = [os.path.join(base_dir, x)
                  for x in tf.io.gfile.listdir(base_dir)]
    return sorted(model_dirs, reverse=True)


def sample_training_examples(state):
    """Sample training examples from recent selfplay games.

    Args:
        state: the RL loop State instance.

    Returns:
        A (num_examples, record_paths) tuple:
         - num_examples : number of examples sampled.
         - record_paths : list of golden chunks up to window_size in length,
                          sorted by path.
    """

    # Read examples from the most recent `window_size` models.
    model_dirs = list_selfplay_dirs(FLAGS.selfplay_dir)[:FLAGS.window_size]
    src_patterns = [os.path.join(x, '*', '*', '*.tfrecord.zz')
                    for x in model_dirs]

    dst_path = os.path.join(FLAGS.golden_chunk_dir,
                            '{}.tfrecord.zz'.format(state.train_model_name))

    logging.info('Writing training chunks to %s', dst_path)
    output = wait(checked_run([
        'bazel-bin/cc/sample_records',
        '--num_read_threads={}'.format(FLAGS.num_read_threads),
        '--num_write_threads={}'.format(FLAGS.num_write_threads),
        '--files_per_pattern={}'.format(FLAGS.min_games_per_iteration),
        '--sample_frac={}'.format(FLAGS.train_filter),
        '--compression=1',
        '--shuffle=true',
        '--dst={}'.format(dst_path)] + src_patterns))

    m = re.search(r"sampled ([\d]+) records", output)
    assert m
    num_examples = int(m.group(1))

    chunk_pattern = os.path.join(
        FLAGS.golden_chunk_dir,
        '{}-*-of-*.tfrecord.zz'.format(state.train_model_name))
    chunk_paths = sorted(tf.gfile.Glob(chunk_pattern))
    assert len(chunk_paths) == FLAGS.num_write_threads

    return (num_examples, chunk_paths)


def append_timestamp(elapsed, model_name):
  # Append the time elapsed from when the RL was started to when this model
  # was trained. GCS files are immutable, so we have to do the append manually.
  timestamps_path = os.path.join(FLAGS.model_dir, 'train_times.txt')
  try:
    with tf.gfile.Open(timestamps_path, 'r') as f:
      timestamps = f.read()
  except tf.errors.NotFoundError:
      timestamps = ''
  timestamps += '{:.3f} {}\n'.format(elapsed, model_name)
  with tf.gfile.Open(timestamps_path, 'w') as f:
      f.write(timestamps)


def train(state):
    """Run training and write a new model to the model_dir.

    Args:
        state: the RL loop State instance.
    """

    wait_for_training_examples(state, FLAGS.min_games_per_iteration)
    num_examples, record_paths = sample_training_examples(state)

    model_path = os.path.join(FLAGS.model_dir, state.train_model_name)

    wait(checked_run([
        'python3', 'train.py',
        '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
        '--work_dir={}'.format(FLAGS.work_dir),
        '--export_path={}'.format(model_path),
        '--use_tpu={}'.format('true' if FLAGS.tpu_name else 'false'),
        '--tpu_name={}'.format(FLAGS.tpu_name),
        '--num_examples={}'.format(num_examples),
        '--freeze=true'] + record_paths))

    # Append the time elapsed from when the RL was started to when this model
    # was trained.
    elapsed = time.time() - state.start_time
    append_timestamp(elapsed, state.train_model_name)

    if FLAGS.validate and state.iter_num - state.start_iter_num > 1:
        try:
            validate(state)
        except Exception as e:
            logging.error(e)


def validate(state):
    src_dirs = list_selfplay_dirs(FLAGS.holdout_dir)[:FLAGS.window_size]

    wait(checked_run([
        'python3', 'validate.py',
        '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'validate.flags')),
        '--work_dir={}'.format(FLAGS.work_dir),
        '--use_tpu={}'.format('true' if FLAGS.tpu_name else 'false'),
        '--tpu_name={}'.format(FLAGS.tpu_name),
        '--expand_validation_dirs'] + src_dirs))


def main(unused_argv):
    """Run the reinforcement learning loop."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                  '%Y-%m-%d %H:%M:%S')

    # MLPerf logging
    mllogger = mllog.get_mllogger()
    mllogger.event(key='cache_clear', value=True)
    mllogger.event(key='init_start', value=None)
    mllogger.event(key='train_batch_size', value=FLAGS.train_batch_size)
    mllogger.event(key='filter_amount', value=FLAGS.filter_amount)
    mllogger.event(key='window_size', value=FLAGS.window_size)
    mllogger.event(
        key='lr_boundaries', value=str(FLAGS.lr_boundaries).strip('[]'))
    mllogger.event(key='lr_rates', value=str(FLAGS.lr_rates).strip('[]'))
    mllogger.event(key='opt_weight_decay', value=FLAGS.l2_strength)
    mllogger.event(
        key='min_selfplay_games_per_generation', value=FLAGS.mlperf_num_games)
    mllogger.event(key='train_samples', value=FLAGS.mlperf_num_games)
    mllogger.event(key='eval_samples', value=FLAGS.mlperf_num_games)
    mllogger.event(key='num_readouts', value=FLAGS.mlperf_num_readouts)
    mllogger.event(
        key='value_init_penalty', value=FLAGS.mlperf_value_init_penalty)
    mllogger.event(key='holdout_pct', value=FLAGS.mlperf_holdout_pct)
    mllogger.event(
        key='disable_resign_pct', value=FLAGS.mlperf_disable_resign_pct)
    mllogger.event(
        key='resign_threshold',
        value=(sum(FLAGS.mlperf_resign_threshold) /
               len(FLAGS.mlperf_resign_threshold)))
    mllogger.event(key='parallel_games', value=FLAGS.mlperf_parallel_games)
    mllogger.event(key='virtual_losses', value=FLAGS.mlperf_virtual_losses)
    mllogger.event(key='gating_win_rate', value=FLAGS.mlperf_gating_win_rate)
    mllogger.event(key='eval_games', value=FLAGS.mlperf_eval_games)

    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # The training loop must be bootstrapped; either by running bootstrap.sh
    # to generate training data from random games, or by running
    # copy_checkpoint.sh to copy an already generated checkpoint.
    iteration_model_names = []
    model_dirs = list_selfplay_dirs(FLAGS.selfplay_dir)
    if not model_dirs:
        raise RuntimeError(
            'Couldn\'t find any selfplay games under %s. Either bootstrap.sh '
            'or init_from_checkpoint.sh must be run before the train loop is '
            'started')
    model_num = int(os.path.basename(model_dirs[0]))

    mllogger.event('init_stop', None)
    mllogger.event('run_start', None)
    with logged_timer('Total time'):
        try:
            state = State(model_num)
            while state.iter_num <= FLAGS.iterations:
                state.iter_num += 1
                iteration_model_names.append(state.train_model_name)
                mllogger.event(
                    key='epoch_start',
                    value=None,
                    metadata={'epoch_num': state.iter_num})
                train(state)
                mllogger.event(
                    key='epoch_stop',
                    value=None,
                    metadata={'epoch_num': state.iter_num})
                mllogger.event(
                    key='save_model',
                    value='{iteration_num: ' + str(state.iter_num) + ' }')
        finally:
                asyncio.get_event_loop().close()

    total_file_count = 0
    for iteration_model_name in iteration_model_names:
        total_file_count = total_file_count + len(
            tf.io.gfile.glob(
                FLAGS.selfplay_dir + '/' + iteration_model_name + '/*/*/*'))

    mllogger.event(
      key='actual_selfplay_games_per_generation',
      value=int(total_file_count / len(iteration_model_names)))


if __name__ == '__main__':
    app.run(main)
