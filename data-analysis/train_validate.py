"""
Utility to train and validate RNN model
"""

import copy
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import time
import torch
from torch.utils.data import DataLoader
from process_tweets import (get_data, convert_tweet2tensor)
from rnn import MyModel
from typing import Union


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(
    epoch: int,
    data_loader: DataLoader,
    model: torch.nn.Module,
    optimizer,
    criterion
):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 2
    cm = torch.zeros(num_class, num_class)
    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        out = model.forward(data)
        loss = criterion(out, target)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss.item(), out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t').format(
                       epoch,
                       idx,
                       len(data_loader),
                       iter_time=iter_time,
                       loss=losses,
                       top1=acc))
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Train Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Train Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm


def validate(
    epoch: int,
    val_loader: DataLoader,
    model: torch.nn.Module,
    criterion
):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 2
    cm = torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        with torch.no_grad():
            out = model.forward(data)
            loss = criterion(out, target)
            
        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss.item(), out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t').format(
                       epoch,
                       idx,
                       len(val_loader),
                       iter_time=iter_time,
                       loss=losses,
                       top1=acc))
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Valid Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Valid Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm


def model_inference(
    model: torch.nn.Module,
    train_file: os.PathLike = None,
    test_file: os.PathLike = None,
    sample_size: int = 10,
):
    _, _, alphabet, longest_sent, _, _ = get_data(train_file, test_file)
    test_df = pd.read_csv(test_file, sep='\t',skiprows=0, encoding = 'utf-8')
    test_sample = test_df.sample(sample_size, random_state=0)
    for idx, row in test_sample.iterrows():
        test_tensor = convert_tweet2tensor(row['text'], alphabet, longest_sent)
        print(f"test text: {test_df.loc[idx,'text']}")
        pred_prob = model.forward(test_tensor)
        pred_label = torch.argmax(pred_prob)
        print(f"true label: {test_df.loc[idx,'HS']}")
        print(f"best model predicted label: {pred_label}")
        print()
