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
import math

#######################################################################################################################################################################

def unpad_input(out_, in_, indices):
    out_[:,:] = in_[indices[:],:]

def pad_input(out_, in_, indices):
    out_[indices[:],:] = in_[:,:]

def unpad_mask(out_, in_, indices):
    out_[:] = in_.flatten()[indices[:]]

#######################################################################################################################################################################

def generate_mask(attention_mask, heads, pad=False, fuse_mask=True):
    seqlen = attention_mask.sum(dim=1).float().cpu()
    if pad == False:
        seqlen[:] = ((seqlen[:] + 16 - 1) / 16).floor()*16
        seqlen[seqlen < 16] = 16
        seqlen = seqlen.int()
        ntokens = seqlen.sum().item()
    else:
        batch = attention_mask.shape[0]
        maxseqlen = attention_mask.shape[1]
        seqlen.fill_(maxseqlen)
        seqlen = seqlen.int()
        ntokens = batch * maxseqlen

    padded_mask = attention_mask.clone()
    for i in range(len(seqlen)):
        padded_mask[i,:seqlen[i]] = 1
    indices = torch.nonzero(padded_mask.flatten(), as_tuple=False).flatten()

    if pad==False and fuse_mask == True:
        mask = torch.zeros([ntokens], device="cuda", dtype=torch.float16)
        unpad_mask(mask, attention_mask, indices)
        mask = (1 - mask) * -10000.0
    elif pad==False and fuse_mask == False:
        padded_mask = (padded_mask.unsqueeze(1) * padded_mask.unsqueeze(2)).unsqueeze(1).half().repeat(1, heads, 1, 1)
        indices_mask = torch.nonzero(padded_mask.flatten(), as_tuple=False).flatten()            
        mask = torch.zeros([len(indices_mask)], device="cuda", dtype=torch.float16)            
        unpad_mask(mask, padded_mask, indices_mask)            
        mask = (1 - mask) * -10000.0
    elif pad==True and fuse_mask == True:
        mask = -10000.0 * (1 - attention_mask).half().view(-1)
    elif pad==True and fuse_mask == False:
        mask = -10000.0 * (1 - (attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2))).unsqueeze(1).half().repeat(1, heads, 1, 1).view(-1)

    return indices, mask, seqlen, ntokens

#######################################################################################################################################################################

class PadInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices, batch, maxseqlen, hidden, ntokens):
        ctx.save_for_backward(indices)
        ctx.hidden = hidden
        ctx.ntokens = ntokens
        ntokens = batch*maxseqlen

        output = torch.zeros([ntokens,hidden], device="cuda", dtype=torch.float16)
        pad_input(output, input, indices)

        return output[:ntokens]

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors

        grad_input = torch.zeros([ctx.ntokens,ctx.hidden], device="cuda", dtype=torch.float16)
        unpad_input(grad_input, grad_output, indices)

        return grad_input[:ctx.ntokens], None, None, None, None, None

#######################################################################################################################################################################

class UnpadInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices, batch, maxseqlen, hidden, ntokens):
        ctx.save_for_backward(indices)
        ctx.hidden = hidden
        ctx.ntokens = batch*maxseqlen

        output = torch.zeros([ntokens, hidden], device="cuda", dtype=torch.float16)
        unpad_input(output, input, indices)

        return output[:ntokens]

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors

        grad_input = torch.zeros([ctx.ntokens,ctx.hidden], device="cuda", dtype=torch.float16)
        pad_input(grad_input, grad_output, indices)

        return grad_input[:ctx.ntokens], None, None, None, None, None

#######################################################################################################################################################################
