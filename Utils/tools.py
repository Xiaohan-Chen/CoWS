import pickle
import torch
import numpy as np
import logging
import time

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def accuracy(outputs, targets):
    """
    Compute the labeled data accuracy.  

    Args:
    - outputs (tensor): model outputs
    - targets (tensor): ground truth

    Return:
    - accuracy (float)
    """
    minibatch_size = len(targets)

    pre = torch.max(outputs.cpu().detach(), 1)[1].numpy()
    y = targets.data.cpu().numpy()

    acc = ((pre == y).sum() / minibatch_size) * 100
    return acc

class AverageMeter(object):
    def __init__(self, name) -> None:
        self.name = name
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
    
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum /self.count
        self.history.append(val)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.delta = delta

    def __call__(self, loss, model, path):
        score = loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model, path)
        elif score > self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model, path)
            self.counter = 0

    def save_checkpoint(self, loss, model, path):
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.loss_min = loss