# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for calculating loss, accuracy, and other model metrics.

Metrics:
 - Padded loss, accuracy, and negative log perplexity. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/metrics.py
 - BLEU approximation. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
 - ROUGE score. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/rouge.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import six
from six.moves import xrange

def _pad_tensors_to_same_length(x, y):
  """Pad x and y so that the results have the same length (second dimension)."""
  x_length = list(x.shape)[1]
  y_length = list(y.shape)[1]

  max_length = x_length if x_length > y_length else y_length

  x = nn.ConstantPad3d((0, 0, 0, max_length - x_length, 0, 0), 0)(x)
  y = nn.ZeroPad2d((0, 0, 0, max_length - y_length))(y)
  return x, y

def softmax_and_cross_entropy(logits, labels):
  return -(labels * F.log_softmax(logits, dim=2)).sum(dim=2)

def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
  """Calculate cross entropy loss while ignoring padding.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
  Returns:
    Returns a float32 tensor with shape
      [batch_size, max(length_logits, length_labels)]
  """
  logits, labels = _pad_tensors_to_same_length(logits, labels)

  with torch.no_grad():
    confidence = 1.0 - smoothing
    low_confidence = (1.0 - confidence) / labels.new_tensor(vocab_size - 1)
    soft_targets = labels.data.new_zeros(*(list(labels.shape)+[vocab_size])) + low_confidence
    soft_targets.scatter_(2, labels.unsqueeze(dim=2).long(), confidence)

  xentropy = softmax_and_cross_entropy(logits, soft_targets)#Variable(soft_targets, requires_grad=False))

  # Calculate the best (lowest) possible value of cross entropy, and
  # subtract from the cross entropy loss.
  normalizing_constant = -(
    confidence * labels.new_tensor(confidence).log() + labels.new_tensor(vocab_size - 1) *
    low_confidence * (low_confidence + 1e-20).log())
  xentropy -= normalizing_constant

  weights = (labels != 0).float()
  return xentropy * weights, weights

def metric_mean(scores, weights, metric_id):
  """ Computes weighted mean and append to the global variable
  in order to remain stateful. This global variable gets cleared by the
  controller outside the scope. We use function's id to uniquely identify
  the metric.
  """
  if metric_id not in global_metrics.keys():
    global_metrics[metric_id] = {'total':0, 'count':0}
    
  weighted_score = (scores * weights).sum()

  global_metrics[metric_id]['total'] += weighted_score.item()
  global_metrics[metric_id]['count'] += weights.sum().item()

  return global_metrics[metric_id]['total'] / global_metrics[metric_id]['count']

def _convert_to_eval_metric(metric_fn):
  """Wrap a metric fn that returns scores and weights as an eval metric fn.

  The input metric_fn returns values for the current batch. The wrapper
  aggregates the return values collected over all of the batches evaluated.

  Args:
    metric_fn: function that returns scores and weights for the current batch's
      logits and predicted labels.

  Returns:
    function that aggregates the scores and weights from metric_fn.
  """
  def problem_metric_fn(*args):
    """Returns an aggregation of the metric_fn's returned values."""
    (scores, weights) = metric_fn(*args)
    metric_id = str(id(metric_fn))
    return metric_mean(scores, weights, metric_id)
  return problem_metric_fn


def get_eval_metrics(logits, labels, params):
  """Return dictionary of model evaluation metrics."""
  metrics = {
      "accuracy": _convert_to_eval_metric(padded_accuracy)(logits, labels),
      "accuracy_top5": _convert_to_eval_metric(padded_accuracy_top5)(
          logits, labels),
      "accuracy_per_sequence": _convert_to_eval_metric(
          padded_sequence_accuracy)(logits, labels),
      "neg_log_perplexity": _convert_to_eval_metric(padded_neg_log_perplexity)(
          logits, labels, params.vocab_size),
      "approx_bleu_score": _convert_to_eval_metric(bleu_score)(logits, labels),
      "rouge_2_fscore": _convert_to_eval_metric(rouge_2_fscore)(logits, labels),
      "rouge_L_fscore": _convert_to_eval_metric(rouge_l_fscore)(logits, labels),
  }
  # Prefix each of the metric names with "metrics/". This allows the metric
  # graphs to display under the "metrics" category in TensorBoard.
  metrics = {"metrics/%s" % k: v for k, v in six.iteritems(metrics)}
  return metrics


def padded_accuracy(logits, labels):
  """Percentage of times that predictions matches labels on non-0s."""
  logits, labels = _pad_tensors_to_same_length(logits, labels)
  weights = (labels != 0).float()
  outputs = (logits.argmax(dim=-1)).int()
  padded_labels = labels.int()
  return (outputs == padded_labels).float(), weights


def padded_accuracy_topk(logits, labels, k):
  """Percentage of times that top-k predictions matches labels on non-0s."""
  logits, labels = _pad_tensors_to_same_length(logits, labels)
  weights = (labels != 0).float()
  effective_k = k if k < logits.shape[-1] else logits.shape[-1]
  _, outputs = torch.topk(logits, k=effective_k)
  outputs = outputs.int()
  padded_labels = labels.int()
  padded_labels = padded_labels.unsqueeze(dim=-1)
  padded_labels = padded_labels.add(outputs.new_zeros(outputs.shape).int())  # Pad to same shape.
  same = (outputs == padded_labels).float()
  same_topk = same.sum(dim=-1)
  return same_topk, weights


def padded_accuracy_top5(logits, labels):
  return padded_accuracy_topk(logits, labels, 5)


def padded_sequence_accuracy(logits, labels):
  """Percentage of times that predictions matches labels everywhere (non-0)."""
  logits, labels = _pad_tensors_to_same_length(logits, labels)
  weights = (labels != 0).float()
  outputs = logits.max(dim=-1)[1].int()
  padded_labels = labels.int()
  not_correct = (outputs != padded_labels).float() * weights
  for dim in range(1, len(outputs.shape)):
    not_correct = not_correct.sum(dim=dim)
  correct_seq = 1.0 - torch.min(labels.new_tensor(1.0).expand_as(not_correct), not_correct)
  return correct_seq, labels.new_tensor(1.0)


def padded_neg_log_perplexity(logits, labels, vocab_size):
  """Average log-perplexity excluding padding 0s. No smoothing."""
  num, den = padded_cross_entropy_loss(logits, labels, 0, vocab_size)
  return -num, den


def bleu_score(logits, labels):
  """Approximate BLEU score computation between labels and predictions.

  An approximate BLEU scoring method since we do not glue word pieces or
  decode the ids and tokenize the output. By default, we use ngram order of 4
  and use brevity penalty. Also, this does not have beam search.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch-size, length_labels]

  Returns:
    bleu: int, approx bleu score
  """
  predictions = logits.max(dim=-1)[1].int()
  bleu = compute_bleu(labels, predictions)
  return bleu, labels.new_tensor(1.0)


def _get_ngrams_with_counter(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in xrange(1, max_order + 1):
    for i in xrange(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
    translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

    overlap = dict((ngram,
                    min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
          ngram]

  precisions = [0] * max_order
  smooth = 1.0

  for i in xrange(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
            i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    ratio = translation_length / reference_length
    bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
  bleu = geo_mean * bp
  return np.float32(bleu)


def rouge_2_fscore(logits, labels):
  """ROUGE-2 F1 score computation between labels and predictions.

  This is an approximate ROUGE scoring method since we do not glue word pieces
  or decode the ids and tokenize the output.

  Args:
    logits: tensor, model predictions
    labels: tensor, gold output.

  Returns:
    rouge2_fscore: approx rouge-2 f1 score.
  """
  predictions = logits.max(dim=-1)[1]
  rouge_2_f_score = rouge_n(predictions, labels)
  return rouge_2_f_score, labels.new_tensor(1.0)


def _get_ngrams(n, text):
  """Calculates n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set


def rouge_n(eval_sentences, ref_sentences, n=2):
  """Computes ROUGE-N f1 score of two text collections of sentences.

  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  Args:
    eval_sentences: Predicted sentences.
    ref_sentences: Sentences from the reference set
    n: Size of ngram.  Defaults to 2.

  Returns:
    f1 score for ROUGE-N
  """
  f1_scores = []
  for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
    eval_ngrams = _get_ngrams(n, eval_sentence)
    ref_ngrams = _get_ngrams(n, ref_sentence)
    ref_count = len(ref_ngrams)
    eval_count = len(eval_ngrams)

    # Count the overlapping ngrams between evaluated and reference
    overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if eval_count == 0:
      precision = 0.0
    else:
      precision = float(overlapping_count) / eval_count
    if ref_count == 0:
      recall = 0.0
    else:
      recall = float(overlapping_count) / ref_count
    f1_scores.append(2.0 * ((precision * recall) / (precision + recall + 1e-8)))

  # return overlapping_count / reference_count
  return np.mean(f1_scores, dtype=np.float32)


def rouge_l_fscore(predictions, labels):
  """ROUGE scores computation between labels and predictions.

  This is an approximate ROUGE scoring method since we do not glue word pieces
  or decode the ids and tokenize the output.

  Args:
    predictions: tensor, model predictions
    labels: tensor, gold output.

  Returns:
    rouge_l_fscore: approx rouge-l f1 score.
  """
  outputs = predictions.max(dim=-1)[1]
  rouge_l_f_score = rouge_l_sentence_level(outputs, labels)
  return rouge_l_f_score, labels.new_tensor(1.0)


def rouge_l_sentence_level(eval_sentences, ref_sentences):
  """Computes ROUGE-L (sentence level) of two collections of sentences.

  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary

  Args:
    eval_sentences: The sentences that have been picked by the summarizer
    ref_sentences: The sentences from the reference set

  Returns:
    A float: F_lcs
  """

  f1_scores = []
  for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
    m = float(len(ref_sentence))
    n = float(len(eval_sentence))
    lcs = _len_lcs(eval_sentence, ref_sentence)
    f1_scores.append(_f_lcs(lcs, m, n))
  return np.mean(f1_scores, dtype=np.float32)


def _len_lcs(x, y):
  """Returns the length of the Longest Common Subsequence between two seqs.

  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns
    integer: Length of LCS between x and y
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  """Computes the length of the LCS between two seqs.

  The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: collection of words
    y: collection of words

  Returns:
    Table of dictionary of coord and len lcs
  """
  y = y.long()
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _f_lcs(llcs, m, n):
  """Computes the LCS-based F-measure score.

  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary

  Returns:
    Float. LCS-based F-measure score
  """
  r_lcs = llcs / m
  p_lcs = llcs / n
  beta = p_lcs / (r_lcs + 1e-12)
  num = (1 + (beta ** 2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta ** 2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs
