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
"""Creates an estimator to train the Transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tempfile
import random
import numpy.random
import subprocess
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

from six.moves import xrange  # pylint: disable=redefined-builtin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import compute_bleu
from utils.dataset import VOCAB_FILE
from model import transformer
from model import model_params
import translate
from utils import dataset
from utils import metrics
from utils import tokenizer


DEFAULT_TRAIN_EPOCHS = 10
BLEU_DIR = "bleu"
INF = 10000
LOG_STEPS = 100

class Estimator():
  """Estimator for controlling/querying model"""
  def __init__(self, model_dir, params):
    if not params.disable_cuda and torch.cuda.is_available():
      self.device = torch.device('cuda')
      self.model = transformer.Transformer(params, self.device).cuda()
    else:
      self.device = torch.device('cpu')
      self.model = transformer.Transformer(params, self.device)

    self.model_dir = model_dir
    self.params = params
    self.set_training_utils(params)

    train_dataset = dataset.TextDataset(params.data_dir, params.train_shards,
            params.max_length, params.batch_size, "train")
    self.train_data_loader = DataLoader(train_dataset,
        num_workers=params.num_cpu_cores)

    eval_dataset = dataset.TextDataset(params.data_dir, params.eval_shards,
            params.max_length, params.batch_size, "eval")
    self.eval_data_loader = DataLoader(eval_dataset,
        num_workers=params.num_cpu_cores)

  def set_training_utils(self, params):
    self.optimizer = optim.Adam(
         params=self.model.parameters(),
         lr=params.learning_rate,
         betas=(params.optimizer_adam_beta1,
                params.optimizer_adam_beta2),
         eps=params.optimizer_adam_epsilon,
        )

    get_learning_rate = lambda step: (params.hidden_size ** -0.5) *\
                        min(1., step/params.learning_rate_warmup_steps) *\
                        torch.rsqrt(torch.tensor(
                                    max(step, params.learning_rate_warmup_steps)
                                    ).float().to(self.device))

    self.scheduler = lr_scheduler.LambdaLR(self.optimizer,
                       lr_lambda=get_learning_rate
                       )

  def train(self, max_steps):

    for steps, batch in enumerate(self.train_data_loader, 0):
      # squeeze out one extra dim added by dataloader
      inputs = batch[0].squeeze(dim=0).to(self.device)
      targets = batch[1].squeeze(dim=0).to(self.device)

      self.scheduler.step()
      self.optimizer.zero_grad()
      logits = self.model(inputs, targets)

      xentropy, weights = metrics.padded_cross_entropy_loss(
                      logits, targets, self.params.label_smoothing,
                      self.params.vocab_size)
      loss = (xentropy * weights).sum() / weights.sum()
      loss.backward()
      self.optimizer.step()

      if steps % LOG_STEPS == 0:
        logging.info("{} steps completed, loss = {:.5f}".format(steps, loss.item()))

      if max_steps is not None and steps == max_steps:
        logging.info("Finished current iteration")
        break
      # For big params, model runs out of memory, hence deleting tensors helps 
      del logits, xentropy, weights, targets, loss

  def evaluate(self):
    metrics.global_metrics = {}
    for i, batch in enumerate(self.eval_data_loader):
      inputs = batch[0].squeeze(dim=0).to(self.device)
      targets = batch[1].squeeze(dim=0).to(self.device)

      with torch.set_grad_enabled(False):
        logits = self.model(inputs, targets, train=False)

      eval_metrics = metrics.get_eval_metrics(logits, targets, self.params)
    return eval_metrics

  def predict(self, inputs):
    """ Returns output dictionary which contains translated text
    """
    with torch.set_grad_enabled(False):
      output_dict = self.model(inputs, train=False)

    return output_dict

def translate_and_compute_bleu(estimator, subtokenizer, bleu_source, bleu_ref):
  """Translate file and report the cased and uncased bleu scores."""
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      estimator, subtokenizer, bleu_source, output_file=tmp_filename,
      print_all_translations=False)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score

def evaluate_and_log_bleu(estimator, bleu_source, bleu_ref):
  """Calculate and record the BLEU score."""
  subtokenizer = tokenizer.Subtokenizer(
      os.path.join(FLAGS.data_dir, FLAGS.vocab_file))

  uncased_score, cased_score = translate_and_compute_bleu(
      estimator, subtokenizer, bleu_source, bleu_ref)

  print("Bleu score (uncased):", uncased_score)
  print("Bleu score (cased):", cased_score)

  return uncased_score, cased_score


def train_schedule(
    estimator, train_eval_iterations, single_iteration_train_steps=None,
    single_iteration_train_epochs=None, bleu_source=None, bleu_ref=None,
    bleu_threshold=None):
  """Train and evaluate model, and optionally compute model's BLEU score.

  **Step vs. Epoch vs. Iteration**

  Steps and epochs are canonical terms used in TensorFlow and general machine
  learning. They are used to describe running a single process (train/eval):
    - Step refers to running the process through a single or batch of examples.
    - Epoch refers to running the process through an entire dataset.

  E.g. training a dataset with 100 examples. The dataset is
  divided into 20 batches with 5 examples per batch. A single training step
  trains the model on one batch. After 20 training steps, the model will have
  trained on every batch in the dataset, or, in other words, one epoch.

  Meanwhile, iteration is used in this implementation to describe running
  multiple processes (training and eval).
    - A single iteration:
      1. trains the model for a specific number of steps or epochs.
      2. evaluates the model.
      3. (if source and ref files are provided) compute BLEU score.

  This function runs through multiple train+eval+bleu iterations.

  Args:
    estimator: Estimator class containing model to train.
    train_eval_iterations: Number of times to repeat the train+eval iteration.
    single_iteration_train_steps: Number of steps to train in one iteration.
    single_iteration_train_epochs: Number of epochs to train in one iteration.
    bleu_source: File containing text to be translated for BLEU calculation.
    bleu_ref: File containing reference translations for BLEU calculation.
    bleu_threshold: minimum BLEU score before training is stopped.

  Raises:
    ValueError: if both or none of single_iteration_train_steps and
      single_iteration_train_epochs were defined.
  """
  # Ensure that exactly one of single_iteration_train_steps and
  # single_iteration_train_epochs is defined.
  if single_iteration_train_steps is None:
    if single_iteration_train_epochs is None:
      raise ValueError(
          "Exactly one of single_iteration_train_steps or "
          "single_iteration_train_epochs must be defined. Both were none.")
  else:
    if single_iteration_train_epochs is not None:
      raise ValueError(
          "Exactly one of single_iteration_train_steps or "
          "single_iteration_train_epochs must be defined. Both were defined.")

  evaluate_bleu = bleu_source is not None and bleu_ref is not None

  # Print out training schedule
  print("Training schedule:")
  if single_iteration_train_epochs is not None:
    print("\t1. Train for %d epochs." % single_iteration_train_epochs)
  else:
    print("\t1. Train for %d steps." % single_iteration_train_steps)
  print("\t2. Evaluate model.")
  if evaluate_bleu:
    print("\t3. Compute BLEU score.")
    if bleu_threshold is not None:
      print("Repeat above steps until the BLEU score reaches", bleu_threshold)
  if not evaluate_bleu or bleu_threshold is None:
    print("Repeat above steps %d times." % train_eval_iterations)

  if evaluate_bleu:
    # Set summary writer to log bleu score.
    if bleu_threshold is not None:
      # Change loop stopping condition if bleu_threshold is defined.
      train_eval_iterations = INF

  # Loop training/evaluation/bleu cycles
  for i in range(train_eval_iterations):
    logging.info("Starting iteration %d", i + 1)

    # Train the model for single_iteration_train_steps or until the input fn
    # runs out of examples (if single_iteration_train_steps is None).
    estimator.train(max_steps=single_iteration_train_steps)

    eval_results = estimator.evaluate()
    logging.info("Evaluation results (iter %d/%d): %s" % (i + 1, train_eval_iterations,
          eval_results))

    if evaluate_bleu:
      uncased_score, _ = evaluate_and_log_bleu(
          estimator, bleu_source, bleu_ref)
      if bleu_threshold is not None and uncased_score > bleu_threshold:
        break


def main(_):

  if FLAGS.params == "base":
    params = model_params.TransformerBaseParams
  elif FLAGS.params == "big":
    params = model_params.TransformerBigParams
  else:
    raise ValueError("Invalid parameter set defined: %s."
                     "Expected 'base' or 'big.'" % FLAGS.params)

  # Determine training schedule based on flags.
  if FLAGS.train_steps is not None and FLAGS.train_epochs is not None:
    raise ValueError("Both --train_steps and --train_epochs were set. Only one "
                     "may be defined.")
  if FLAGS.train_steps is not None:
    train_eval_iterations = FLAGS.train_steps // FLAGS.steps_between_eval
    single_iteration_train_steps = FLAGS.steps_between_eval
    single_iteration_train_epochs = None
  else:
    if FLAGS.train_epochs is None:
      FLAGS.train_epochs = DEFAULT_TRAIN_EPOCHS
    train_eval_iterations = FLAGS.train_epochs // FLAGS.epochs_between_eval
    single_iteration_train_steps = None
    single_iteration_train_epochs = FLAGS.epochs_between_eval

  # Make sure that the BLEU source and ref files if set
  if FLAGS.bleu_source is not None and FLAGS.bleu_ref is not None:
    if not os.path.exists(FLAGS.bleu_source):
      raise ValueError("BLEU source file %s does not exist" % FLAGS.bleu_source)
    if not os.path.exists(FLAGS.bleu_ref):
      raise ValueError("BLEU source file %s does not exist" % FLAGS.bleu_ref)

  # Add flag-defined parameters to params object
  params.data_dir = FLAGS.data_dir
  params.num_cpu_cores = FLAGS.num_cpu_cores
  params.disable_cuda = FLAGS.disable_cuda
  params.epochs_between_eval = FLAGS.epochs_between_eval
  params.repeat_dataset = single_iteration_train_epochs
  #TODO change these vars to find number of shards based on file pattern
  params.train_shards = 100
  params.eval_shards = 1

  estimator = Estimator(model_dir=FLAGS.model_dir, params=params)
  train_schedule(
       estimator, train_eval_iterations, single_iteration_train_steps,
       single_iteration_train_epochs, FLAGS.bleu_source, FLAGS.bleu_ref,
       FLAGS.bleu_threshold)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_dir", "-dd", type=str, default="/tmp/translate_ende",
      help="[default: %(default)s] Directory containing training and "
           "evaluation data, and vocab file used for encoding.",
      metavar="<DD>")
  parser.add_argument(
      "--vocab_file", "-vf", type=str, default=VOCAB_FILE,
      help="[default: %(default)s] Name of vocabulary file.",
      metavar="<vf>")
  parser.add_argument(
      "--model_dir", "-md", type=str, default="/tmp/transformer_model",
      help="[default: %(default)s] Directory to save Transformer model "
           "training checkpoints",
      metavar="<MD>")
  parser.add_argument(
      "--params", "-p", type=str, default="big", choices=["base", "big"],
      help="[default: %(default)s] Parameter set to use when creating and "
           "training the model.",
      metavar="<P>")
  parser.add_argument(
      "--num_cpu_cores", "-nc", type=int, default=4,
      help="[default: %(default)s] Number of CPU cores to use in the input "
           "pipeline.",
      metavar="<NC>")
  parser.add_argument(
      "--disable_cuda", "-dc", action='store_true',
      help="Disable CUDA to run the model on CPU",
      )

  # Flags for training with epochs. (default)
  parser.add_argument(
      "--train_epochs", "-te", type=int, default=None,
      help="The number of epochs used to train. If both --train_epochs and "
           "--train_steps are not set, the model will train for %d epochs." %
      DEFAULT_TRAIN_EPOCHS,
      metavar="<TE>")
  parser.add_argument(
      "--epochs_between_eval", "-ebe", type=int, default=1,
      help="[default: %(default)s] The number of training epochs to run "
           "between evaluations.",
      metavar="<TE>")

  # Flags for training with steps (may be used for debugging)
  parser.add_argument(
      "--train_steps", "-ts", type=int, default=None,
      help="Total number of training steps. If both --train_epochs and "
           "--train_steps are not set, the model will train for %d epochs." %
      DEFAULT_TRAIN_EPOCHS,
      metavar="<TS>")
  parser.add_argument(
      "--steps_between_eval", "-sbe", type=int, default=1000,
      help="[default: %(default)s] Number of training steps to run between "
           "evaluations.",
      metavar="<SBE>")

  # BLEU score computation
  parser.add_argument(
      "--bleu_source", "-bs", type=str, default=None,
      help="Path to source file containing text translate when calculating the "
           "official BLEU score. Both --bleu_source and --bleu_ref must be "
           "set. The BLEU score will be calculated during model evaluation.",
      metavar="<BS>")
  parser.add_argument(
      "--bleu_ref", "-br", type=str, default=None,
      help="Path to file containing the reference translation for calculating "
           "the official BLEU score. Both --bleu_source and --bleu_ref must be "
           "set. The BLEU score will be calculated during model evaluation.",
      metavar="<BR>")
  parser.add_argument(
      "--bleu_threshold", "-bt", type=float, default=None,
      help="Stop training when the uncased BLEU score reaches this value. "
           "Setting this overrides the total number of steps or epochs set by "
           "--train_steps or --train_epochs.",
      metavar="<BT>")


  parser.add_argument(
      "--random_seed", "-rs", type=int, default=None,
      help="the random seed to use", metavar="<SEED>")

  FLAGS, unparsed = parser.parse_known_args()
  logging.info('Setting random seed = %d', FLAGS.random_seed)
  if FLAGS.random_seed is None:
    raise Exception('No Random seed given')
  seed = FLAGS.random_seed
  random.seed(seed)
  numpy.random.seed(seed)
  # Works safely for non-cuda system as well
  torch.cuda.manual_seed(seed)


  main(sys.argv)
