"""
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
"""

import gc
import logging
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import *

import Configs.train as parms
from Preparedata.load_data import load_data, load_big_data, DatasetAugWaveSpec
from Model import ResNet1D, ResNet2D, Attention
from Utils.logger import setlogger
from Utils import tools

def loaddata(args):
    train_datadict, _, _ = load_data(args)
    # train_datadict = load_big_data()

    train_dataset = DatasetAugWaveSpec(train_datadict)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,\
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return train_loader

class Model(nn.Module):
    def __init__(self, dim=32) -> None:
        super().__init__()
        self.backbone_wave = ResNet1D.resnet50_1D()
        self.backbone_wave.fc = nn.Identity()

        self.backbone_spec = ResNet2D.resnet50_2D()
        self.backbone_spec.fc = nn.Identity()

        self.temperature = nn.Parameter(torch.tensor(1.))

        self.projection_wave = Attention.Attention(512, dim, 0.5)
        self.projection_spec = Attention.Attention(512, dim, 0.5)

    def _cosin_sim(self, input1, input2):
        """Cosine simularity
        """
        input1 = F.normalize(input1, p = 2, dim = -1)
        input2 = F.normalize(input2, p = 2, dim = -1)

        temp = self.temperature.exp()

        similarity = input1 @ input2.T * temp
        return similarity
    
    def _criterion(self, input):
        batch_size = input.size(0)
        target = torch.arange(batch_size)
        device = input.device
        target = target.to(device)

        criterion = nn.CrossEntropyLoss()

        loss_ts = criterion(input, target)
        loss_st = criterion(input.T, target)

        loss = (loss_ts + loss_st) / 2.

        return loss
    
    def forward(self, wave, spec):
        wave = self.backbone_wave(wave)
        spec = self.backbone_spec(spec)

        wave = self.projection_wave(wave)
        spec = self.projection_spec(spec)

        logits = self._cosin_sim(wave, spec)
        loss = self._criterion(logits)

        return loss

def train(model, train_loader, optimizer, device):
    model.train()
    losses = tools.AverageMeter("train_loss")
    with tqdm(total=len(train_loader), ncols=70, leave=False) as pbar:
        for i, (wave, spec) in enumerate(train_loader):

            wave, spec = wave.to(device), spec.to(device)
            loss = model(wave, spec)

            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update()
    
    return losses.avg

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = loaddata(args)
    model = Model(dim=args.dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)

    early_stopping = tools.EarlyStopping(args.patience)
    logging.info(">>>>> CoWS Pre-training ...")
    for epoch in range(args.max_epochs):
        train_loss = train(model, train_loader, optimizer, device)

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if epoch % args.print_step == 0:
            logging.info(f"Epoch: {epoch+1:>4}/{args.max_epochs}, train loss: {train_loss:.6f}")

        path = "./Checkpoint/"+"CoWS_"+args.dataset+"_load"+str(args.load)+".pth"
        early_stopping(train_loss, model.backbone_wave, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    args = parms.parse_args()

    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger("./logs/CoWS.log")

    if not os.path.exists("./Checkpoint"):
        os.makedirs("./Checkpoint")

    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    
    starttime = time.time()
    main(args)
    endtime = time.time()
    logging.info(f"Total time: {endtime - starttime:.2f}s")