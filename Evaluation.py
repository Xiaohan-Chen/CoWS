"""
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
"""

import logging
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import *

import Configs.evaluation as parms
from Preparedata.load_data import load_data, BaseDataset, label_data_indices
from Model import ResNet1D
from Utils.logger import setlogger
from Utils import tools

def loaddata(args):
    train_datadict, val_datadict, test_datadict = load_data(args)

    train_dataset = BaseDataset(train_datadict)
    val_dataset = BaseDataset(val_datadict)
    test_dataset = BaseDataset(test_datadict)

    labeled_indices = label_data_indices(args)
    sampler = torch.utils.data.SubsetRandomSampler(labeled_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,\
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, \
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, \
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader


def load_checkpoint(model, path, mode, random_initial=False):
    if not random_initial:
        checkpoint = torch.load(path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        assert missing_keys == ["fc.weight", "fc.bias"]
    model.fc.weight.data.normal_(mean=0.0, std=0.1)
    model.fc.bias.data.zero_()

    if mode == "linear_protocol":
        model.requires_grad_(False)
        model.fc.requires_grad_(True)

    return model

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    lossmeter = tools.AverageMeter("train_loss")
    accmeter = tools.AverageMeter("train_acc")
    with tqdm(total=len(dataloader), ncols=70, leave=False) as pbar:
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y.long())
            acc = tools.accuracy(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossmeter.update(loss.item())
            accmeter.update(acc)

            pbar.update()

    return accmeter.avg, lossmeter.avg

def evaluate(model, dataloader, criterion, device):
    model.eval()
    lossmeter = tools.AverageMeter("train_loss")
    accmeter = tools.AverageMeter("train_acc")

    with torch.no_grad():
        for i, (x,y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y.long())
            acc = tools.accuracy(output, y)

            lossmeter.update(loss.item())
            accmeter.update(acc)

    return accmeter.avg, lossmeter.avg


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = loaddata(args)

    classes = len(args.labels.split(","))
    model = ResNet1D.resnet50_1D(num_classes=classes).to(device)

    model_name, data_name, load = args.model, args.dataset, str(args.load)
    checkpoint_path = os.path.join("./Checkpoint", model_name+"_"+data_name+"_"+"load"+load+".pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist, please pre-train the model first.")
    model = load_checkpoint(model, checkpoint_path, args.mode)

    classifier_parameters, backbone_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            backbone_parameters.append(param)

    criterion = nn.CrossEntropyLoss()
    param_groups = [dict(params=classifier_parameters, lr=args.classifier_lr)]
    if args.mode != "linear_protocol":
        param_groups.append(dict(params=backbone_parameters, lr=args.backbone_lr))
    optimizer = torch.optim.SGD(param_groups, 0, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)

    best_acc = 0.0
    for epoch in range(args.max_epochs):
        train_acc, train_loss = train(model, train_loader, optimizer, criterion, device)
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        lr_scheduler.step()

        if val_acc >= best_acc:
            best_acc = val_acc
            test_acc, _ = evaluate(model, test_loader, criterion, device)
        
        if args.verbose:
            logging.info(f"Epoch: {epoch+1:>4}/{args.max_epochs}, train_acc: {train_acc:6.2f}%, " 
            f"train_loss: {train_loss:.4f}, val_acc: {val_acc:6.2f}%, val_loss: {val_loss:.4f}")
    logging.info(f">>> Test {args.model} acc: {test_acc:7.2f}%")

if __name__ == "__main__":
    args = parms.parse_args()
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger("./logs/Evaluation.log")
    logging.info(f">>>>> {args.mode} evaluation ...")
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    main(args)