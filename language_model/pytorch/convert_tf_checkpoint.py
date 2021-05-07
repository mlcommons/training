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
import argparse

from modeling import BertForPretraining, BertConfig

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument('--tf_checkpoint',
                        type=str,
                        default="/google_bert_data",
                        help="Path to directory containing TF checkpoint")
    parser.add_argument('--bert_config_path',
                        type=str,
                        default="/workspace/phase1",
                        help="Path bert_config.json is located in")
    parser.add_argument('--output_checkpoint', type=str,
                        default='./checkpoint.pt',
                        help="Path to output PyT checkpoint")

    return parser.parse_args()

def prepare_model(args, device):

    # Prepare model
    config = BertConfig.from_json_file(args.bert_config_path)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
        print('padded vocab size to: {}'.format(config.vocab_size))

    # Set some options that the config file is expected to have (but don't need to be set properly
    # at this point)
    config.pad = False
    config.unpad = False
    config.dense_seq_output = False
    config.fused_mha = False
    config.fused_gelu_bias = False
    config.fuse_qkv = False
    config.fuse_scale = False
    config.fuse_mask = False
    config.fuse_dropout = False
    config.apex_softmax = False
    config.enable_stream = False
    if config.fuse_mask == True: config.apex_softmax = True
    if config.pad == False: config.enable_stream = True
    if config.unpad == True: config.fused_mha = False

    #Load from TF checkpoint
    model = BertForPretraining.from_pretrained(args.tf_checkpoint, from_tf=True, config=config)

    return model

def main():
    args = parse_arguments()
    device = torch.device("cuda")

    model = prepare_model(args, device)

    torch.save({'model' : model.state_dict() }, args.output_checkpoint)


if __name__ == "__main__":
    main()

