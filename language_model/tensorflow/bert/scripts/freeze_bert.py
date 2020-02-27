"""
Script to add Placeholder as inputs to a Bert checkpoint
"""

import argparse
import contextlib
import json
import os
import tempfile
from enum import Enum

import logging

logger = logging.getLogger(__name__)
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

import modeling


def get_args():
    parser = argparse.ArgumentParser(description='Start a BertServer for serving')
    parser.add_argument('--checkpoint', type=str, required=True, help='filename of the checkpoint file')
    parser.add_argument('--config', type=str, default='bert_config.json', help='fconfig file for BERT model.')
    return parser.parse_args()


def optimize_graph(args):
    config = tf.ConfigProto()
    model_dir = os.path.dirname(args.checkpoint)

    init_checkpoint = os.path.join(model_dir, args.checkpoint)
    with tf.gfile.GFile(os.path.join(model_dir, args.config), 'r') as f:
        bert_config = modeling.BertConfig.from_dict(json.load(f))

    MAX_SEQ_LENGTH, BS = None, None
    input_ids = tf.placeholder(tf.int32, (BS, MAX_SEQ_LENGTH), 'input_ids')
    input_mask = tf.placeholder(tf.int32, (BS, MAX_SEQ_LENGTH), 'input_mask')
    input_type_ids = tf.placeholder(tf.int32, (BS, MAX_SEQ_LENGTH), 'input_type_ids')
    input_tensors = [input_ids, input_mask, input_type_ids]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=False)

    tvars = tf.trainable_variables()

    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    pooled = model.get_pooled_output()
    pooled = tf.identity(pooled, 'final_encodes')
    output_tensors = [pooled]
    graph_def = tf.get_default_graph().as_graph_def()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dtypes = [n.dtype for n in input_tensors]
        input_node_names = [n.name[:-2] for n in input_tensors]
        output_node_names = [n.name[:-2] for n in output_tensors]
        graph_def = optimize_for_inference(
            graph_def, input_node_names, output_node_names,
            [dtype.as_datatype_enum for dtype in dtypes],
            False)
        graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names)

    pb_file = args.checkpoint + ".pb"
    with tf.gfile.GFile(pb_file, 'wb') as f:
        f.write(graph_def.SerializeToString())

    return pb_file, bert_config


def main():
    args = get_args()
    optimize_graph(args)


if __name__ == "__main__":
    main()
