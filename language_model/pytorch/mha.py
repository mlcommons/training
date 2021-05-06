# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from apex.contrib.multihead_attn import fast_mask_softmax_dropout_func

from bmm1 import *
from bmm2 import *
from padding import *
from softmax import *

class FastUnpadBertSelfAttention(nn.Module):
    def __init__(self, config, enable_stream=True, enable_sync=True, fuse_mask=True, fuse_scale=True, fuse_qkv=True, fuse_dropout=True, apex_softmax=True, pad=True):
        super(FastUnpadBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        self.fuse_qkv = fuse_qkv
        self.fuse_scale = fuse_scale
        self.fuse_mask = fuse_mask
        self.fuse_dropout = fuse_dropout
        self.apex_softmax = apex_softmax
        self.pad = pad
        self.enable_stream = enable_stream

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.fuse_qkv:
            self.bmm1 = Bmm1Strided(None,None,self.num_attention_heads,self.attention_head_size, scale=self.fuse_scale, stream=enable_stream, sync=enable_sync, timer=False)
            self.bmm2 = Bmm2Strided(None,None,self.num_attention_heads,self.attention_head_size, stream=enable_stream, sync=enable_sync, timer=False)
        else:
            self.bmm1 = Bmm1(None,None,self.num_attention_heads,self.attention_head_size, scale=self.fuse_scale, stream=enable_stream, sync=enable_sync)
            self.bmm2 = Bmm2(None,None,self.num_attention_heads,self.attention_head_size, stream=enable_stream, sync=enable_sync)

        if self.fuse_dropout == False:
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if self.fuse_mask == True and self.fuse_dropout == True:
            self.softmax = FastMaskSoftmaxDropout(dim=-1, dropout_prob=config.attention_probs_dropout_prob,stream=enable_stream, sync=(not self.pad), timer=False)
        elif self.fuse_mask == True:
            self.softmax = FastMaskSoftmax(dim=-1, stream=enable_stream, sync=enable_sync, timer=False)
        else:
            self.softmax = FastSoftmax(dim=-1, stream=enable_stream, sync=enable_sync, timer=False)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 3, 1)

    def pytorch_softmax(self,attention_scores, batch, seqlen, heads):
        ntokens2 = 0
        for i in range(batch):
            ntokens2 += seqlen[i]*seqlen[i]*self.num_attention_heads
        attention_probs = torch.zeros(ntokens2, device="cuda", dtype=torch.float16)
        ntokens2 = 0
        for i in range(batch):
            tokens2 = seqlen[i]*seqlen[i]*self.num_attention_heads
            attention_probs[ntokens2:ntokens2+tokens2] = F.softmax(attention_scores[ntokens2:ntokens2+tokens2].view(1,self.num_attention_heads,seqlen[i],seqlen[i]), dim=-1).flatten().contiguous()
            ntokens2 += tokens2
        return attention_probs

    def forward(self, hidden_states, attention_mask, seqlen, batch, is_training=True):

        self.batch = batch

        # QKV
        if self.fuse_qkv:
            weight = torch.cat([self.query.weight.view(self.num_attention_heads,self.attention_head_size,1,self.hidden_size), self.key.weight.view(self.num_attention_heads,self.attention_head_size,1,self.hidden_size), self.value.weight.view(self.num_attention_heads,self.attention_head_size,1,self.hidden_size)], dim=1).reshape(self.all_head_size*3,self.hidden_size).contiguous()
            bias = torch.cat([self.query.bias.view(self.num_attention_heads,1,self.attention_head_size), self.key.bias.view(self.num_attention_heads,1,self.attention_head_size), self.value.bias.view(self.num_attention_heads,1,self.attention_head_size)],dim=1).reshape(3*self.hidden_size).contiguous()
            mixed_x_layer = torch.addmm(bias, hidden_states, weight.t())
        else:
            query_layer = self.query(hidden_states)
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)            

        # BMM1.
        if self.enable_stream: torch.cuda.synchronize()
        if self.fuse_qkv:
            attention_scores, qkv_layer = self.bmm1(mixed_x_layer, self.batch, seqlen)
        else:
            attention_scores = self.bmm1(query_layer, key_layer, self.batch, seqlen)            

        if self.enable_stream: torch.cuda.synchronize()
        if self.fuse_scale == False:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Softmax.
        if self.enable_stream: torch.cuda.synchronize()        
        if self.fuse_mask ==True and self.fuse_dropout == True:
            attention_probs = self.softmax(attention_scores, attention_mask, self.batch, seqlen, self.num_attention_heads, is_training)
        elif self.fuse_mask == True:
            attention_probs = self.softmax(attention_scores, attention_mask, self.batch, seqlen, self.num_attention_heads)
        else:
            attention_scores = attention_scores + attention_mask.view(-1)
            if self.apex_softmax == True:
                attention_probs = self.softmax(attention_scores, self.batch, seqlen, self.num_attention_heads)
            else:
                if self.pad == True:
                    attention_probs = F.softmax(attention_scores.view(batch,self.num_attention_heads,seqlen[0],seqlen[0]), dim=-1).flatten().contiguous()
                else:
                    attention_probs = self.pytorch_softmax(attention_scores, self.batch, seqlen, self.num_attention_heads)

        # Dropout.
        if self.enable_stream: torch.cuda.synchronize()                
        if self.fuse_dropout == False:
            attention_probs = self.dropout(attention_probs)

        # BMM2.
        if self.enable_stream: torch.cuda.synchronize()
        if self.fuse_qkv:
            context_layer = self.bmm2(attention_probs, qkv_layer, self.batch, seqlen)
        else:
            context_layer = self.bmm2(attention_probs, value_layer, self.batch, seqlen)

        if self.enable_stream: torch.cuda.synchronize()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = torch.reshape(context_layer, new_context_layer_shape)
        return context_layer
