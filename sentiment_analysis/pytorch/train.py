import argparse
import warnings
from torchtext import data
from torchtext import datasets
import torch
import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def IMDB_dataset(use_cuda=True, batch_size=128, max_len=2470):
    """
    Pytorch generator for IMDB dataset.
    Args:
    use_cuda - bool
    batch_size - int
    max_len - int - max length of the sentence in train.
    All smaller sentences will be padded to have length = max_len.
    All larger sentences will be cropped.
    Returns:
    train_iter, test_iter - batch generators
    len(TEXT.vocab) - vocabulary size. Necessary for the embedding layer
    batch_size
    """
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


class ConvNet(nn.Module):
    """
    Architecture:
    Embedding layer with customizable vocab size and
    embeding size,
    For filter size 3 and 4:
    2d Convolutional layer,
    2d Max Pooling layer,
    Fully-connected layer applied for
    concatinaned outputs of two separated convotional
    layers
    """

    def __init__(self, vocab_size, embedding_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.conv_3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=4),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=4),
        )
        self.fc = nn.Linear(436560, 2)

    def forward(self, sentences):
        embed_layer = nn.Embedding(
            self.vocab_size,
            self.embedding_size,
            sparse=True)
        sentences_embedded = embed_layer(sentences)
        sentences_embedded = sentences_embedded.unsqueeze(1)
        out_3 = self.conv_3(sentences_embedded)
        out_4 = self.conv_4(sentences_embedded)
        out_3 = out_3.view(out_3.size(0), -1)
        out_4 = out_4.view(out_4.size(0), -1)
        out = torch.cat((out_3, out_4), 1)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)


def train(
        model,
        optimizer,
        n_epochs,
        train_iter,
        test_iter,
        vocab_size,
        batch_size,
        quality,
        train_size):
    """
    Performs training with n_epochs steps.
    train_iter - torch iterator over the train set
    test_test - torch iterator over the test set
    vocab_size - int, number of unique words to be embedded
    batch_size - int, size of minibatch
    quality - float, accuracy to reach on test set
    train_size - int, number of sentences in train set
    """
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_iter)
        val_loss, val_acc = test(model, test_iter, quality)
        train_log.extend(train_loss)
        train_acc_log.extend(train_acc)
        steps = train_size // batch_size
        val_log.append((steps * (epoch + 1), np.mean(val_loss)))
        average_val_acc = np.mean(val_acc)
        val_acc_log.append((steps * (epoch + 1), average_val_acc))
        print("Epoch =", epoch,
              ", train-accuracy =", np.mean(train_acc), ", train-loss =",
              np.mean(train_loss), ", validation-accuracy =", average_val_acc,
              ", validation-loss =", np.mean(val_loss))
        if average_val_acc > target_val_acc:
            break

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
        acc_log.append(acc)
        loss = loss.data[0]
        loss_log.append(loss)
    return loss_log, acc_log


def main(use_cuda, seed, quality, embedding_size, train_size):
    if use_cuda and not torch.cuda.is_available():
        warnings.warn(
            "CUDA device is not accessible! Setting use_cuda to False.")
        use_cuda = False
    
    train_iter, test_iter, vocab_size, batch_size = IMDB_dataset(use_cuda)
    model = ConvNet(vocab_size, embedding_size)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    n_epochs = 3

    train(
        model,
        optimizer,
        n_epochs,
        train_iter,
        test_iter,
        vocab_size,
        batch_size,
        quality,
        train_size)


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
    parser.add_argument('-e', '--embedding_size', type=int, required=False,
                        default=1024, help="Length of embedding vector")

    args = parser.parse_args()

    if args.model == 'conv':
        main(use_cuda=True,  # Runs on CPU if "False"
             seed=args.seed,
             quality=args.target_quality,
             embedding_size=args.embedding_size,
             train_size=25000)
    else:
        raise NotImplementedError
