import os
import heapq
import math
import time
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import multiprocessing as mp

import utils
from neumf import NeuMF
from dataset import CFTrainDataset, load_test_ratings, load_test_negs
from convert import (TEST_NEG_FILENAME, TEST_RATINGS_FILENAME,
                     TRAIN_RATINGS_FILENAME)


def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('data', type=str,
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of examples for each iteration')
    parser.add_argument('-f', '--factors', type=int, default=8,
                        help='number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[64, 32, 16, 8],
                        help='size of hidden layers for MLP')
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float,
                        help='stop training early at threshold')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes for evaluating model')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of workers for training DataLoader')
    return parser.parse_args()


def predict(model, users, items, batch_size=1024, use_cuda=True):
    batches = [(users[i:i + batch_size], items[i:i + batch_size])
               for i in range(0, len(users), batch_size)]
    preds = []
    for user, item in batches:
        def proc(x):
            x = np.array(x)
            x = torch.from_numpy(x)
            if use_cuda:
                x = x.cuda(async=True)
            return torch.autograd.Variable(x)
        outp = model(proc(user), proc(item), sigmoid=True)
        outp = outp.data.cpu().numpy()
        preds += list(outp.flatten())
    return preds


def _calculate_hit(ranked, test_item):
    return int(test_item in ranked)


def _calculate_ndcg(ranked, test_item):
    for i, item in enumerate(ranked):
        if item == test_item:
            return math.log(2) / math.log(i + 2)
    return 0.


def eval_one(rating, items, model, K, use_cuda=True):
    user = rating[0]
    test_item = rating[1]
    items.append(test_item)
    users = [user] * len(items)
    predictions = predict(model, users, items, use_cuda=use_cuda)

    map_item_score = {item: pred for item, pred in zip(items, predictions)}
    ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    hit = _calculate_hit(ranked, test_item)
    ndcg = _calculate_ndcg(ranked, test_item)
    return hit, ndcg


def val_epoch(model, ratings, negs, K, use_cuda=True, output=None, epoch=None,
              processes=1):
    if epoch is None:
        print("Initial evaluation")
    else:
        print("Epoch {} evaluation".format(epoch))
    start = datetime.now()
    model.eval()
    if processes > 1:
        context = mp.get_context('spawn')
        _eval_one = partial(eval_one, model=model, K=K, use_cuda=use_cuda)
        with context.Pool(processes=processes) as workers:
            hits_and_ndcg = workers.starmap(_eval_one, zip(ratings, negs))
        hits, ndcgs = zip(*hits_and_ndcg)
    else:
        hits, ndcgs = [], []
        for rating, items in zip(ratings, negs):
            hit, ndcg = eval_one(rating, items, model, K, use_cuda=use_cuda)
            hits.append(hit)
            ndcgs.append(ndcg)

    hits = np.array(hits, dtype=np.float32)
    ndcgs = np.array(ndcgs, dtype=np.float32)

    end = datetime.now()
    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = np.mean(hits)
        result['NDCG'] = np.mean(ndcgs)
        utils.save_result(result, output)

    return hits, ndcgs


def main():
    args = parse_args()
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    # Save configuration to file
    config = {k: v for k, v in args.__dict__.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    config['local_timestamp'] = str(datetime.now())
    run_dir = "./run/neumf/{}".format(config['timestamp'])
    print("Saving config and results to {}".format(run_dir))
    if not os.path.exists(run_dir) and run_dir != '':
        os.makedirs(run_dir)
    utils.save_config(config, run_dir)

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    t1 = time.time()
    # Load Data
    print('Loading data')
    train_dataset = CFTrainDataset(
        os.path.join(args.data, TRAIN_RATINGS_FILENAME), args.negative_samples)
    train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    test_ratings = load_test_ratings(os.path.join(args.data, TEST_RATINGS_FILENAME))  # noqa: E501
    test_negs = load_test_negs(os.path.join(args.data, TEST_NEG_FILENAME))
    nb_users, nb_items = train_dataset.nb_users, train_dataset.nb_items
    print('Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d'
          % (time.time()-t1, nb_users, nb_items, train_dataset.mat.nnz,
             len(test_ratings)))

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors, mf_reg=0.,
                  mlp_layer_sizes=args.layers,
                  mlp_layer_regs=[0. for i in args.layers])
    print(model)
    print("{} parameters".format(utils.count_parameters(model)))

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    # Add optimizer and loss to graph
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    if use_cuda:
        # Move model and loss to GPU
        model = model.cuda()
        criterion = criterion.cuda()

    # Create files for tracking training
    valid_results_file = os.path.join(run_dir, 'valid_results.csv')

    # Calculate initial Hit Ratio and NDCG
    hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                            use_cuda=use_cuda, processes=args.processes)
    print('Initial HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}'
          .format(K=args.topk, hit_rate=np.mean(hits), ndcg=np.mean(ndcgs)))
    for epoch in range(args.epochs):
        model.train()
        losses = utils.AverageMeter()

        begin = time.time()
        loader = tqdm.tqdm(train_dataloader)
        for batch_index, (user, item, label) in enumerate(loader):
            user = torch.autograd.Variable(user, requires_grad=False)
            item = torch.autograd.Variable(item, requires_grad=False)
            label = torch.autograd.Variable(label, requires_grad=False)
            if use_cuda:
                user = user.cuda(async=True)
                item = item.cuda(async=True)
                label = label.cuda(async=True)

            outputs = model(user, item)
            loss = criterion(outputs, label)
            losses.update(loss.data.item(), user.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save stats to file
            description = ('Epoch {} Loss {loss.val:.4f} ({loss.avg:.4f})'
                           .format(epoch, loss=losses))
            loader.set_description(description)

        train_time = time.time() - begin
        begin = time.time()
        hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                                use_cuda=use_cuda, output=valid_results_file,
                                epoch=epoch, processes=args.processes)
        val_time = time.time() - begin
        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
              ' train_time = {train_time:.2f}, val_time = {val_time:.2f}'
              .format(epoch=epoch, K=args.topk, hit_rate=np.mean(hits),
                      ndcg=np.mean(ndcgs), train_time=train_time,
                      val_time=val_time))
        if args.threshold is not None:
            if np.mean(hits) >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                return 0


if __name__ == '__main__':
    main()
