import torch.nn as nn
from mlperf_compliance import mlperf_log

import seq2seq.data.config as config
from seq2seq.models.decoder import ResidualRecurrentDecoder
from seq2seq.models.encoder import ResidualRecurrentEncoder
from seq2seq.models.seq2seq_base import Seq2Seq
from seq2seq.utils import gnmt_print


class GNMT(Seq2Seq):
    """
    GNMT v2 model
    """
    def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2,
                 batch_first=False, share_embedding=True):
        """
        Constructor for the GNMT v2 model.

        :param vocab_size: size of vocabulary (number of tokens)
        :param hidden_size: internal hidden size of the model
        :param num_layers: number of layers, applies to both encoder and
            decoder
        :param dropout: probability of dropout (in encoder and decoder)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param share_embedding: if True embeddings are shared between encoder
            and decoder
        """

        super(GNMT, self).__init__(batch_first=batch_first)

        gnmt_print(key=mlperf_log.MODEL_HP_NUM_LAYERS,
                   value=num_layers, sync=False)
        gnmt_print(key=mlperf_log.MODEL_HP_HIDDEN_SIZE,
                   value=hidden_size, sync=False)
        gnmt_print(key=mlperf_log.MODEL_HP_DROPOUT,
                   value=dropout, sync=False)

        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_size,
                                    padding_idx=config.PAD)
            nn.init.uniform_(embedder.weight.data, -0.1, 0.1)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)

        self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)

    def forward(self, input_encoder, input_enc_len, input_decoder):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)

        return output
