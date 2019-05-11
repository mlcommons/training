import torch.jit
import os
import math
import time
import timeit
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser
from alias_generator import AliasSample
import pickle
from convert import generate_negatives
from convert import generate_negatives_flat
from convert import CACHE_FN

import tqdm
import numpy as np
import torch
import torch.nn as nn

import utils
from neumf import NeuMF

from mlperf_compliance import mlperf_log

def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('data', type=str,
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of examples for each iteration')
    parser.add_argument('--valid-batch-size', type=int, default=2**20,
                        help='number of examples in each validation chunk')
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
    parser.add_argument('--valid-negative', type=int, default=999,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes for evaluating model')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of workers for training DataLoader')
    parser.add_argument('--beta1', '-b1', type=float, default=0.9,
                        help='beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float, default=0.999,
                        help='beta1 for Adam')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps for Adam')
    parser.add_argument('--user_scaling', default=1, type=int)
    parser.add_argument('--item_scaling', default=1, type=int)
    parser.add_argument('--cpu_dataloader', action='store_true',
                        help='pre-process data on cpu to save memory')
    parser.add_argument('--random_negatives', action='store_true',
                        help='do not check train negatives for existence in dataset')
    return parser.parse_args()


# TODO: val_epoch is not currently supported on cpu
def val_epoch(model, x, y, dup_mask, real_indices, K, samples_per_user, num_user, output=None,
              epoch=None, loss=None):

    start = datetime.now()
    log_2 = math.log(2)

    model.eval()
    hits = torch.tensor(0., device='cuda')
    ndcg = torch.tensor(0., device='cuda')

    with torch.no_grad():
        for i, (u,n) in enumerate(zip(x,y)):
            res = model(u.cuda().view(-1), n.cuda().view(-1), sigmoid=True).detach().view(-1,samples_per_user)
            # set duplicate results for the same item to -1 before topk
            res[dup_mask[i]] = -1
            out = torch.topk(res,K)[1]
            # topk in pytorch is stable(if not sort)
            # key(item):value(predicetion) pairs are ordered as original key(item) order
            # so we need the first position of real item(stored in real_indices) to check if it is in topk
            ifzero = (out == real_indices[i].cuda().view(-1,1))
            hits += ifzero.sum()
            ndcg += (log_2 / (torch.nonzero(ifzero)[:,1].view(-1).to(torch.float)+2).log_()).sum()

    mlperf_log.ncf_print(key=mlperf_log.EVAL_SIZE, value={"epoch": epoch, "value": num_user * samples_per_user})
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_USERS, value=num_user)
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_NEG, value=samples_per_user - 1)

    end = datetime.now()

    hits = hits.item()
    ndcg = ndcg.item()

    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = hits/num_user
        result['NDCG'] = ndcg/num_user
        result['loss'] = loss
        utils.save_result(result, output)

    return hits/num_user, ndcg/num_user


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
    # Check where to put data loader
    if use_cuda:
        dataloader_device = 'cpu' if args.cpu_dataloader else 'cuda'
    else:
        dataloader_device = 'cpu'

    # more like load trigger timmer now
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_NUM_EVAL, value=args.valid_negative)
    # The default of np.random.choice is replace=True, so does pytorch random_()
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_SAMPLE_EVAL_REPLACEMENT, value=True)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT, value=True)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_EVAL_NEG_GEN)

    # sync worker before timing.
    torch.cuda.synchronize()

    #===========================================================================
    #== The clock starts on loading the preprocessed data. =====================
    #===========================================================================
    mlperf_log.ncf_print(key=mlperf_log.RUN_START)
    run_start_time = time.time()

    print(datetime.now(), "Loading test ratings.")
    test_ratings = [torch.LongTensor()] * args.user_scaling

    for chunk in range(args.user_scaling):
        test_ratings[chunk] = torch.from_numpy(np.load(args.data + '/testx' 
                + str(args.user_scaling) + 'x' + str(args.item_scaling) 
                + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0'])
        
    fn_prefix = args.data + '/' + CACHE_FN.format(args.user_scaling, args.item_scaling)
    sampler_cache = fn_prefix + "cached_sampler.pkl"
    print(datetime.now(), "Loading preprocessed sampler.")
    if os.path.exists(args.data):
      print("Using alias file: {}".format(args.data))
      with open(sampler_cache, "rb") as f:
        sampler, pos_users, pos_items, nb_items, _ = pickle.load(f)
    print(datetime.now(), "Alias table loaded.")

    nb_users = len(sampler.num_regions)
    train_users = torch.from_numpy(pos_users).type(torch.LongTensor)
    train_items = torch.from_numpy(pos_items).type(torch.LongTensor)

    mlperf_log.ncf_print(key=mlperf_log.INPUT_SIZE, value=len(train_users))
    # produce things not change between epoch
    # mask for filtering duplicates with real sample
    # note: test data is removed before create mask, same as reference
    # create label
    train_label = torch.ones_like(train_users, dtype=torch.float32)
    neg_label = torch.zeros_like(train_label, dtype=torch.float32)
    neg_label = neg_label.repeat(args.negative_samples)
    train_label = torch.cat((train_label,neg_label))
    del neg_label

    test_pos = [l[:,1].reshape(-1,1) for l in test_ratings]
    test_negatives = [torch.LongTensor()] * args.user_scaling
    test_neg_items = [torch.LongTensor()] * args.user_scaling
    
    print(datetime.now(), "Loading test negatives.")
    for chunk in range(args.user_scaling):
        file_name = (args.data + '/test_negx' + str(args.user_scaling) + 'x'
                + str(args.item_scaling) + '_' + str(chunk) + '.npz')
        raw_data = np.load(file_name, encoding='bytes')
        test_negatives[chunk] = torch.from_numpy(raw_data['arr_0'])
        print(datetime.now(), "Test negative chunk {} of {} loaded ({} users).".format(
              chunk+1, args.user_scaling, test_negatives[chunk].size()))

    test_neg_items = [l[:, 1] for l in test_negatives]

    # create items with real sample at last position
    test_items = [torch.cat((a.reshape(-1,args.valid_negative), b), dim=1)
            for a, b in zip(test_neg_items, test_pos)]
    del test_ratings, test_neg_items

    # generate dup mask and real indice for exact same behavior on duplication compare to reference
    # here we need a sort that is stable(keep order of duplicates)
    # this is a version works on integer
    sorted_items, indices = zip(*[torch.sort(l) for l in test_items]) # [1,1,1,2], [3,1,0,2]
    sum_item_indices = [a.float()+b.float()/len(b[0]) 
            for a, b in zip(sorted_items, indices)] #[1.75,1.25,1.0,2.5]
    indices_order = [torch.sort(l)[1] for l in sum_item_indices] #[2,1,0,3]
    stable_indices = [torch.gather(a, 1, b) 
            for a, b in zip(indices, indices_order)] #[0,1,3,2]
    # produce -1 mask
    dup_mask = [(l[:,0:-1] == l[:,1:]) for l in sorted_items]
    dup_mask = [torch.cat((torch.zeros_like(a, dtype=torch.uint8), b),dim=1)
            for a, b in zip(test_pos, dup_mask)]
    dup_mask = [torch.gather(a,1,b.sort()[1])
            for a, b in zip(dup_mask, stable_indices)]
    # produce real sample indices to later check in topk
    sorted_items, indices = zip(*[(a != b).sort()
            for a, b in zip(test_items, test_pos)])
    sum_item_indices = [(a.float()) + (b.float())/len(b[0])
            for a, b in zip(sorted_items, indices)]
    indices_order = [torch.sort(l)[1] for l in sum_item_indices]
    stable_indices = [torch.gather(a, 1, b)
            for a, b in zip(indices, indices_order)]
    real_indices = [l[:, 0] for l in stable_indices]
    del sorted_items, indices, sum_item_indices, indices_order, stable_indices, test_pos

    # For our dataset, test set is identical to user set, so arange() provides
    # all test users.
    test_users = torch.arange(nb_users, dtype=torch.long)
    test_users = test_users[:, None]
    test_users = test_users + torch.zeros(1+args.valid_negative, dtype=torch.long)
    # test_items needs to be of type Long in order to be used in embedding
    test_items = torch.cat(test_items).type(torch.long)

    dup_mask = torch.cat(dup_mask)
    real_indices = torch.cat(real_indices)

    # make pytorch memory behavior more consistent later
    torch.cuda.empty_cache()

    mlperf_log.ncf_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_ORDER)  # we shuffled later with randperm

    print(datetime.now(),
        "Data loading done {:.1f} sec. #user={}, #item={}, #train={}, #test={}".format(
          time.time()-run_start_time, nb_users, nb_items, len(train_users), nb_users))

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
    params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
    criterion = nn.BCEWithLogitsLoss(reduction = 'none') # use torch.mean() with dim later to avoid copy to host
    mlperf_log.ncf_print(key=mlperf_log.OPT_LR, value=args.learning_rate)
    mlperf_log.ncf_print(key=mlperf_log.OPT_NAME, value="Adam")
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=args.beta1)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=args.beta2)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=args.eps)
    mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_LOSS_FN, value=mlperf_log.BCE)

    if use_cuda:
        # Move model and loss to GPU
        model = model.cuda()
        criterion = criterion.cuda()

    local_batch = args.batch_size
    traced_criterion = torch.jit.trace(criterion.forward, (torch.rand(local_batch,1),torch.rand(local_batch,1)))

    # Create files for tracking training
    valid_results_file = os.path.join(run_dir, 'valid_results.csv')

    # Calculate initial Hit Ratio and NDCG
    samples_per_user = test_items.size(1)
    users_per_valid_batch = args.valid_batch_size // samples_per_user

    test_users = test_users.split(users_per_valid_batch)
    test_items = test_items.split(users_per_valid_batch)
    dup_mask = dup_mask.split(users_per_valid_batch)
    real_indices = real_indices.split(users_per_valid_batch)

    hr, ndcg = val_epoch(model, test_users, test_items, dup_mask, real_indices, args.topk, samples_per_user=samples_per_user,
                         num_user=nb_users)
    print('Initial HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}'
          .format(K=args.topk, hit_rate=hr, ndcg=ndcg))
    success = False
    mlperf_log.ncf_print(key=mlperf_log.TRAIN_LOOP)
    for epoch in range(args.epochs):

        mlperf_log.ncf_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_NUM_NEG, value=args.negative_samples)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN)
        begin = time.time()
        
        st = timeit.default_timer()
        if args.random_negatives:
            neg_users = train_users.repeat(args.negative_samples)
            neg_items = torch.empty_like(neg_users, dtype=torch.int64).random_(0, nb_items)
        else:
            negatives = generate_negatives(
                sampler,
                args.negative_samples,
                train_users.numpy())
            negatives = torch.from_numpy(negatives)
            neg_users = negatives[:, 0]
            neg_items = negatives[:, 1]

        print("generate_negatives loop time: {:.2f}", timeit.default_timer() - st)

        after_neg_gen = time.time()

        st = timeit.default_timer()
        epoch_users = torch.cat((train_users,neg_users))
        epoch_items = torch.cat((train_items,neg_items))
        del neg_users, neg_items

        # shuffle prepared data and split into batches
        epoch_indices = torch.randperm(len(epoch_users), device=dataloader_device)
        epoch_size = len(epoch_indices)
        epoch_users = epoch_users[epoch_indices]
        epoch_items = epoch_items[epoch_indices]
        epoch_label = train_label[epoch_indices]
        epoch_users_list = epoch_users.split(local_batch)
        epoch_items_list = epoch_items.split(local_batch)
        epoch_label_list = epoch_label.split(local_batch)

        print("shuffle time: {:.2f}", timeit.default_timer() - st)

        # only print progress bar on rank 0
        num_batches = (epoch_size + args.batch_size - 1) // args.batch_size
        qbar = tqdm.tqdm(range(num_batches))
        # handle extremely rare case where last batch size < number of worker
        if len(epoch_users_list) < num_batches:
            print("epoch_size % batch_size < number of worker!")
            exit(1)
        
        after_shuffle = time.time()
        
        neg_gen_time = (after_neg_gen - begin)
        shuffle_time = (after_shuffle - after_neg_gen)

        for i in qbar:
            # selecting input from prepared data
            user = epoch_users_list[i].cuda()
            item = epoch_items_list[i].cuda()
            label = epoch_label_list[i].view(-1,1).cuda()

            for p in model.parameters():
                p.grad = None

            outputs = model(user, item)
            loss = traced_criterion(outputs, label).float()
            loss = torch.mean(loss.view(-1), 0)

            loss.backward()
            optimizer.step()
       
        del epoch_users, epoch_items, epoch_label, epoch_users_list, epoch_items_list, epoch_label_list, user, item, label
        train_time = time.time() - begin
        begin = time.time()

        mlperf_log.ncf_print(key=mlperf_log.EVAL_START, value=epoch)

        hr, ndcg = val_epoch(model, test_users, test_items, dup_mask, real_indices, args.topk, samples_per_user=samples_per_user,
                             num_user=nb_users, output=valid_results_file, epoch=epoch, loss=loss.data.item())

        val_time = time.time() - begin
        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
                ' train_time = {train_time:.2f}, val_time = {val_time:.2f}, loss = {loss:.4f},'
                ' neg_gen: {neg_gen_time:.4f}, shuffle_time: {shuffle_time:.2f}'
              .format(epoch=epoch, K=args.topk, hit_rate=hr,
                      ndcg=ndcg, train_time=train_time,
                      val_time=val_time, loss=loss.data.item(),
                      neg_gen_time=neg_gen_time, shuffle_time=shuffle_time))

        mlperf_log.ncf_print(key=mlperf_log.EVAL_ACCURACY, value={"epoch": epoch, "value": hr})
        mlperf_log.ncf_print(key=mlperf_log.EVAL_TARGET, value=args.threshold)
        mlperf_log.ncf_print(key=mlperf_log.EVAL_STOP, value=epoch)

        if args.threshold is not None:
            if hr >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                success = True
                break

    mlperf_log.ncf_print(key=mlperf_log.RUN_STOP, value={"success": success})
    run_stop_time = time.time()
    mlperf_log.ncf_print(key=mlperf_log.RUN_FINAL)

    # easy way of tracking mlperf score
    if success:
        print("mlperf_score", run_stop_time - run_start_time)

if __name__ == '__main__':
    main()
