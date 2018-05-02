from __future__ import print_function

import unittest
import paddle.fluid as fluid
import paddle
import contextlib
import math
import numpy as np
import sys
import os
import random
import argparse

# Define the architecture for the convolution model
def convolution_net(data, label, input_dim, class_dim=2, emb_dim=1024,
                    hid_dim=1024):
    # First layer is an embedding layer, which serves as a lookup table for
    # a one-hot encoding vector.
    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)

    # Output of embedding layer is fed to 2 independent sequence_conv_pool
    # layers. The sequence_conv_pool layer is a combination of a sequence- 
    # convolution layer and a sequence pooling layer
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        act="elu",
        pool_type="max")
    conv_4 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4,
        act="elu",
        pool_type="max")

    prediction = fluid.layers.fc(input=[conv_3, conv_4],
                                 size=class_dim,
                                 act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, accuracy, prediction


# Define the architecture for the LSTM model
def stacked_lstm_net(data,
                     label,
                     input_dim,
                     class_dim=2,
                     emb_dim=256,
                     hid_dim=1024,
                     stacked_num=3):
    assert stacked_num % 2 == 1

    # First layer is again an embedding layer
    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)

    # The embedding layer is followed by `stacked_num` FC+LSTM layers
    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = fluid.layers.fc(input=[fc_last, lstm_last],
                                 size=class_dim,
                                 act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, accuracy, prediction


def train(word_dict,
          net_method,
          use_cuda,
          seed,
          quality,
          save_dirname=None):
    BATCH_SIZE = 128
    PASS_NUM = 100
    dict_dim = len(word_dict)
    class_dim = 2
    target_val_acc = quality

    # Seed for batch producer
    random.seed(seed) 
    
    # Seed for weight initialization
    fluid.default_startup_program().random_seed = seed

    # Setup input features and label as data layers
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    cost, acc_out, prediction = net_method(
        data, label, input_dim=dict_dim, class_dim=class_dim)

    # Initialize a test program for obtaining test accuracy and cost
    test_program = fluid.default_main_program().clone(for_test=True)

    # Setup Adam optimizer
    adam = fluid.optimizer.Adam(learning_rate=0.0005) #Learning rate of 5e-4 works for conv models and 2e-3 for LSTM model

    optimize_ops, params_grads = adam.minimize(cost)

    # Create reader to iterate over training set
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=25000),
        batch_size=BATCH_SIZE)

    # Setup place and executor for runtime
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
    
    # Create reader to iterate over validation set
    test_reader = paddle.batch(
                    paddle.dataset.imdb.test(word_dict), batch_size=BATCH_SIZE)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())

        for pass_id in xrange(PASS_NUM):
            train_loss_set = []
            train_acc_set = []  
   
            # Calculate average training loss and accuracy
            # across all mini-batches in the training set
            for batch_id, data in enumerate(train_reader()):
                cost_val, acc_val = exe.run(main_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[cost, acc_out])
		train_loss_set.append(float(cost_val))
		train_acc_set.append(float(acc_val)) 
	    train_loss = np.array(train_loss_set).mean()
            train_acc = np.array(train_acc_set).mean() * 100

            # Calculate average valication loss and accuracy 
            # across all mini-batches in the validation set
            acc_set = []
            avg_loss_set = []
            for tid, test_data in enumerate(test_reader()):
                avg_loss_np, acc_np = exe.run(
                            program=test_program,
                            feed=feeder.feed(test_data),
                            fetch_list=[cost, acc_out])
                acc_set.append(float(acc_np))
                avg_loss_set.append(float(avg_loss_np))
            acc_val = np.array(acc_set).mean() * 100 
            avg_loss_val = np.array(avg_loss_set).mean()
            print("Epoch =", pass_id, ", train-accuracy =", train_acc, ", train-loss =", train_loss, ", validation-accuracy =", acc_val, ", validation-loss =", avg_loss_val)

            if acc_val > target_val_acc:
                ## Exit the program on reaching desired accuracy value
                break

    train_loop(fluid.default_main_program())


def main(word_dict, net_method, use_cuda,
         seed, quality, save_dirname=None):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        print("Paddle isn't compiled with CUDA!")
        return

    train(
        word_dict,
        net_method,
        use_cuda,
        seed,
        quality,
        save_dirname=save_dirname)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Parse arguments
    parser.add_argument('-m', '--model', nargs='?', required=False,
                        choices=['conv', 'lstm'], default='conv',
                        help="Model type for sentiment analysis")
    parser.add_argument('-q', '--target_quality', type=float, required=False,
                        default=90.6,
                        help="Target validation quality to stop training")
    parser.add_argument('-s', '--seed', type=int, required=False, default=1,
                        help="Seed for random number generator")

    word_dict = paddle.dataset.imdb.word_dict()
    args = parser.parse_args()

    if args.model == 'conv':
        main(word_dict,
             net_method=convolution_net,
             use_cuda=True, # Runs on CPU if "False"
             seed=args.seed,
             quality=args.target_quality,
             save_dirname="understand_sentiment_conv.inference.model")
    else:
        main(word_dict,
             use_method=stacked_lstm_net,
             use_cuda=True,
             seed=args.seed,
             quality=args.target_quality,
             save_dirname="understand_sentiment_lstm.inference.model")
