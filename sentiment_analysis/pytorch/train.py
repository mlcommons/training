from torchtext import data
from torchtext import datasets
import torch
import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def IMDB_dataset(use_cuda=True, batch_size=128, max_len=2470):
    device = "cuda:0" if use_cuda else -1
    # set up fields
    TEXT = data.Field(lower=True, fix_length=max_len, batch_first=True)
    LABEL = data.Field(sequential=False)
    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)
    # build the vocabulary
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_size=batch_size, device=device)
    return train_iter, test_iter, len(TEXT.vocab), batch_size


def train(model, optimizer, n_epochs, train_iter, test_iter, vocab_size, batch_size, quality):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []


    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, opt, train_iter)
        
        val_loss, val_acc = test(model, test_iter, quality)
        
        train_log.extend(train_loss)
        train_acc_log.extend(train_acc)

        steps = 25000 / batch_size
        val_log.append((steps * (epoch + 1), np.mean(val_loss)))
        val_acc_log.append((steps * (epoch + 1), np.mean(val_acc)))
        
        plot_history(train_log, val_log)
        plot_history(train_acc_log, val_acc_log, title='accuracy') 
    print("Final error: {:.2%}".format(1 - val_acc_log[-1][1]))

def train_epoch(model, optimizer, train_iter):
    loss_log, acc_log = [], []
    model.train()
    for batch in tqdm.tqdm(train_iter):
        data = Variable(batch.text)
        target = Variable(batch.label) - 1
        optimizer.zero_grad()
        output = model(data)
        pred = torch.max(output, 1)[1].data.numpy()
        acc = np.mean(pred == target.data.numpy())
        acc_log.append(acc)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        loss = loss.data[0]
        loss_log.append(loss)
    return loss_log, acc_log

def test(model, test_iter, quality):
    loss_log, acc_log = [], []
    model.eval()
    for batch in tqdm.tqdm(test_iter):
        data = Variable(batch.text)
        target = Variable(batch.label) - 1
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = torch.max(output, 1)[1].data.numpy()
        acc = np.mean(pred == target.data.numpy())
        #if acc >= quality:
        #    break
        acc_log.append(acc)
        loss = loss.data[0]
        loss_log.append(loss)
    return loss_log, acc_log

def plot_history(train_history, val_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)
    points = np.array(val_history)
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')
    plt.legend(loc='best')
    plt.grid()
    plt.show()



def main(model, use_cuda, seed, quality):
    if use_cuda and not torch.cuda.is_available():
        warnings.warn("CUDA device is not accessible! Setting use_cuda to False.")
        use_cuda = False

    train_iter, test_iter, vocab_size, batch_size = IMDB_dataset(use_cuda)
    train(model, optimizer, n_epochs, train_iter, test_iter, vocab_size, batch_size, quality)



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

    args = parser.parse_args()

    print (args)

    main(use_cuda=True, # Runs on CPU if "False"
         seed=args.seed,
         quality=args.target_quality)