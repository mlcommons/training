import math
from collections import OrderedDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}

supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchLogSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            batch_size = input_.size()[0]
            return torch.stack([F.log_softmax(input_[i]) for i in range(batch_size)], 0)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True, rnn_activation='tanh',
        hidden_threshold=-100,
        qh_i=0, qh_f=0,
        qi_i = 0, qi_f = 0,
        bias = False,
        bn_factor  = False,
        bn_bias    = False,
        bn_weights = False):
        super(BatchRNN, self).__init__()
        self.bn_factor = bn_factor
        self.bn_bias = bn_bias
        self.bn_weights = bn_weights

        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.bidirectional  = bidirectional
        self.rnn_activation = rnn_activation

        if batch_norm:
          self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size))
        else:
          self.batch_norm = None

        self.hidden_threshold = hidden_threshold
        self.qh_i = qh_i
        self.qh_f = qh_f
        self.qi_i = qi_i
        self.qi_f = qi_f
        self.bias = bias


        if rnn_type == nn.GRU or rnn_type == nn.LSTM:
          self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                              bidirectional=bidirectional, bias=bias)
        else:
          # Use RNN with relu or tanh non-linearity
          self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                              bidirectional=bidirectional, bias=bias,
                              nonlinearity=rnn_activation)

        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
            x, _ = self.rnn(x)
        else:
            x, _ = self.rnn(x)

        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        self.rnn.flatten_parameters()
        return x


class DeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=5, audio_conf=None,
                 bidirectional=True, distillation=False, rnn_activation='tanh',
                 bias = False,
                 hidden_threshold = 0,
                 qh_i = 0, qh_f = 0,
                 qi_i = 0, qi_f = 0,
                 bn_factor  = False,
                 bn_bias    = False,
                 bn_weights = False):
        super(DeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}

        self._version       = '0.0.1'
        self._hidden_size   = rnn_hidden_size
        self._hidden_layers = nb_layers
        self._rnn_type      = rnn_type
        self._audio_conf    = audio_conf or {}
        self._labels        = labels
        self.rnn_activation = rnn_activation
        self.hidden_threshold = hidden_threshold
        self.qh_i = qh_i
        self.qh_f = qh_f
        self.bn_factor = bn_factor
        self.bn_bias = bn_bias
        self.bn_weights = bn_weights

        sample_rate = self._audio_conf.get("sample_rate", 16000)
        window_size = self._audio_conf.get("window_size", 0.02)
        num_classes = len(self._labels)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional,
                       batch_norm=False,
                       rnn_activation=rnn_activation,
                       hidden_threshold = hidden_threshold,
                       qh_i = qh_i,
                       qh_f = qh_f,
                       qi_i = qi_i,
                       qi_f = qi_f,
                       bias = bias,
                       bn_factor  = bn_factor,
                       bn_weights = bn_weights,
                       bn_bias    = bn_bias)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
          rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                         bidirectional=bidirectional,
                         batch_norm=False,
                         rnn_activation=rnn_activation,
                         hidden_threshold = hidden_threshold,
                         qh_i = qh_i,
                         qh_f = qh_f,
                         qi_i = qi_i,
                         qi_f = qi_f,
                         bias = bias,
                         bn_factor  = bn_factor,
                         bn_weights = bn_weights,
                         bn_bias    = bn_bias)
          rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_log_softmax = InferenceBatchLogSoftmax()

    def forward(self, x):
        x = self.conv(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        x = self.rnns(x)

        x = self.fc(x)
        x = x.transpose(0, 1)

        # identity in training mode, logsoftmax in eval mode
        x = self.inference_log_softmax(x)

        return x

    @classmethod
    def load_model(cls, path, cuda=False, rnn_type = 'gru', rnn_activation='tanh',
                   hidden_threshold = 0,
                   qh_i = 0, qh_f = 0,
                   qi_i = 0, qi_f = 0,
                   bias = False,
                   bn_factor  = False,
                   bn_bias    = False,
                   bn_weights = False):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[rnn_type], rnn_activation=rnn_activation,
                    hidden_threshold = hidden_threshold,
                    qh_i = qh_i,
                    qh_f = qh_f,
                    qi_i = qi_i,
                    qi_f = qi_f,
                    bias = bias,
                    bn_factor  = bn_factor,
                    bn_bias    = bn_bias,
                    bn_weights = bn_weights)
        model.load_state_dict(package['state_dict'])
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        model_is_cuda = next(model.parameters()).is_cuda
        model = model.module if model_is_cuda else model
        package = {
            'version': model._version,
            'hidden_size': model._hidden_size,
            'hidden_layers': model._hidden_layers,
            'rnn_type': supported_rnns_inv.get(model._rnn_type, model._rnn_type.__name__.lower()),
            'audio_conf': model._audio_conf,
            'labels': model._labels,
            'state_dict': model.state_dict()
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_audio_conf(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._audio_conf if model_is_cuda else model._audio_conf
