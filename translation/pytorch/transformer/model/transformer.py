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
"""Defines the Transformer model, and its encoder and decoder stacks.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Ported in PyTorch, original Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import attention_layer
from model import beam_search
from model import embedding_layer
from model import ffn_layer
from model import model_utils
from utils.tokenizer import EOS_ID

# Define defaults for parameters
_NEG_INF = -1e9

def init_weights(m):
  if type(m) == nn.Linear:
    nn.init.xavier_uniform_(m.weight)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.01)

class Transformer(nn.Module):
  """Transformer model for sequence to sequence data.

  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self, params, device):
    """Initialize layers to build Transformer model.

    Args:
      params: hyperparameter object defining layer sizes, dropout values, etc.
      train: boolean indicating whether the model is in training mode. Used to
        determine if dropout layers should be added.
    """
    super(Transformer, self).__init__()
    self.params = params
    self.device = device

    self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
        params.vocab_size, params.hidden_size)

    self.encoder_stack = EncoderStack(params)
    self.decoder_stack = DecoderStack(params)

    self.apply(init_weights)

  def forward(self, inputs, targets=None, train=True):
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      targets: None or int tensor with shape [batch_size, target_length].

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          output: [batch_size, decoded length]
          score: [batch_size, float]}
    """
    # Calculate attention bias for encoder self-attention and decoder
    # multi-headed attention layers.
    attention_bias = model_utils.get_padding_bias(inputs)
    # Run the inputs through the encoder layer to map the symbol
    # representations to continuous representations.
    encoder_outputs = self.encode(inputs, attention_bias)

    # Generate output sequence if targets is None, or return logits if target
    # sequence is known.
    if targets is None:
      return self.predict(encoder_outputs, attention_bias)
    else:
      logits = self.decode(targets, encoder_outputs, attention_bias)
      return logits

  def encode(self, inputs, attention_bias, train=True):
    """Generate continuous representation for inputs.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    # Prepare inputs to the layer stack by adding positional encodings and
    # applying dropout.
    embedded_inputs = self.embedding_softmax_layer(inputs)
    inputs_padding = model_utils.get_padding(inputs)

    length = inputs.new_tensor(embedded_inputs.size()[1])
    pos_encoding = model_utils.get_position_encoding(
        length, self.params.hidden_size).to(self.device)
    encoder_inputs = embedded_inputs + pos_encoding

    if train:
      encoder_inputs = F.dropout(encoder_inputs, p=self.params.layer_postprocess_dropout, training=True)

    return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

  def decode(self, targets, encoder_outputs, attention_bias, train=True):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence.
        int tensor with shape [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence.
        float tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    # Prepare inputs to decoder layers by shifting targets, adding positional
    # encoding and applying dropout.
    decoder_inputs = self.embedding_softmax_layer(targets)
    # Shift targets to the right, and remove the last element
    pad = (0, 0, 1, 0)
    decoder_inputs = F.pad(
            decoder_inputs, pad)[:, :-1, :]
    length = targets.new_tensor(decoder_inputs.size()[1])
    decoder_inputs += model_utils.get_position_encoding(
          length, self.params.hidden_size)
    if train:
      decoder_inputs = F.dropout(decoder_inputs, p=self.params.layer_postprocess_dropout, training=True)

    # Run values
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
          length)
    outputs = self.decoder_stack(
          decoder_inputs, encoder_outputs, decoder_self_attention_bias,
          attention_bias)
    logits = self.embedding_softmax_layer.linear(outputs)
    return logits

  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""

    timing_signal = model_utils.get_position_encoding(
        max_decode_length + 1, self.params.hidden_size)
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        max_decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences.
          int tensor with shape [batch_size * beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      i = i.int()
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_softmax_layer(decoder_input)
      decoder_input += timing_signal[i:i + 1]

      self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
      decoder_outputs = self.decoder_stack(
          decoder_input, cache.get("encoder_outputs"), self_attention_bias,
          cache.get("encoder_decoder_attention_bias"), cache)
      logits = self.embedding_softmax_layer.linear(decoder_outputs)
      logits = torch.squeeze(logits, 1)
      return logits, cache
    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias):
    """Return predicted sequence."""
    batch_size = encoder_outputs.new_tensor(encoder_outputs.size()[0]).int()
    input_length = encoder_outputs.new_tensor(encoder_outputs.size()[1])
    max_decode_length = input_length + self.params.extra_decode_length

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = batch_size.new_zeros(int(batch_size))

    # Create cache storing decoder attention values for each layer.
    cache = {
        "layer_%d" % layer: {
            "k": batch_size.new_zeros((batch_size, 0, self.params.hidden_size)),
            "v": batch_size.new_zeros((batch_size, 0, self.params.hidden_size)),
        } for layer in range(self.params.num_hidden_layers)}

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self.params.vocab_size,
        beam_size=self.params.beam_size,
        alpha=self.params.alpha,
        max_decode_length=max_decode_length,
        eos_id=EOS_ID)

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    return {"outputs": top_decoded_ids, "scores": top_scores}


class LayerNormalization(nn.Module):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size
    self.scale = nn.Parameter(torch.ones(self.hidden_size))
    self.bias = nn.Parameter(torch.zeros(self.hidden_size))

  def forward(self, x, epsilon=1e-6, train=True):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.mean(torch.mul(x - mean, x - mean), dim=-1, keepdim=True)
    norm_x = (x - mean) * torch.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(nn.Module):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self.postprocess_dropout = params.layer_postprocess_dropout

    # Create normalization layer
    self.layer_norm = LayerNormalization(params.hidden_size)

  def forward(self, x, *args, **kwargs):
    # Preprocessing: apply layer normalization
    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if kwargs['train']:
      y = F.dropout(y, training=True, p=self.postprocess_dropout)
    return x + y


class EncoderStack(nn.Module):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params):
    super(EncoderStack, self).__init__()
    self.layers = nn.ModuleList()
    for _ in range(params.num_hidden_layers):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params.hidden_size, params.num_heads, params.attention_dropout)
      feed_forward_network = ffn_layer.FeedFowardNetwork(
          params.hidden_size, params.filter_size, params.relu_dropout)

      self.layers.append(nn.ModuleList([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)]))

    # Create final layer normalization layer.
    self.output_normalization = LayerNormalization(params.hidden_size)

  def forward(self, encoder_inputs, attention_bias, inputs_padding, train=True):
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      encoder_inputs = self_attention_layer(encoder_inputs, attention_bias, train=train)
      encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding, train=train)

    return self.output_normalization(encoder_inputs)


class DecoderStack(nn.Module):
  """Transformer decoder stack.

  Like the encoder stack, the decoder stack is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self, params):
    super(DecoderStack, self).__init__()
    self.layers = nn.ModuleList()
    for _ in range(params.num_hidden_layers):
      self_attention_layer = attention_layer.SelfAttention(
          params.hidden_size, params.num_heads, params.attention_dropout)
      enc_dec_attention_layer = attention_layer.Attention(
          params.hidden_size, params.num_heads, params.attention_dropout)
      feed_forward_network = ffn_layer.FeedFowardNetwork(
          params.hidden_size, params.filter_size, params.relu_dropout)

      self.layers.append(nn.ModuleList([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(enc_dec_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)]))

    self.output_normalization = LayerNormalization(params.hidden_size)

  def forward(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
           attention_bias, cache=None, train=True):
    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None

      decoder_inputs = self_attention_layer(
          decoder_inputs, decoder_self_attention_bias, cache=layer_cache, train=train)

      decoder_inputs = enc_dec_attention_layer(
          decoder_inputs, encoder_outputs, attention_bias, train=train)

      decoder_inputs = feed_forward_network(decoder_inputs, train=train)

    return self.output_normalization(decoder_inputs)
