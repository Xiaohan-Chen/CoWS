import os
import math
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from Preparedata.Augmentation import *
import time

class BaseDataset(Dataset):
    def __init__(self, dataset: dict) -> None:
        super().__init__()
        self.dataset = dataset
        self.classes = len(dataset)
        self.x, self.y = self._read(dataset)

    def _read(self, dataset):
        x = np.concatenate([dataset[key] for key in dataset.keys()])
        y = []
        for i, key in enumerate(dataset.keys()):
            number = len(dataset[key])
            y.append(np.tile(i, number))
        y = np.concatenate(y)
        x = x[:, np.newaxis, :]
        return x, y
    
    def __len__(self):
        count = 0
        for key in self.dataset.keys():
            count += len(self.dataset[key])
        return count
    
    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)

        return data, label

class DatasetAugWave(BaseDataset):
    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)

        wave1 = random_waveform_transform(data)
        wave2 = random_waveform_transform(data)

        return wave1, wave2

class DatasetAugTimeFreq(BaseDataset):
    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)

        wave = random_waveform_transform(data)
        freq = frequency_transforms(data)

        return wave, freq

class DatasetAugWaveSpec(BaseDataset):
    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)

        wave = random_waveform_transform(data)
        spec = random_spectrogram_transform(data)

        spec = torch.cat((spec, spec, spec), 0)

        return wave, spec

def load_from_pkl(dataname, load):
    datadir = os.path.join("./Data", dataname)
    datapath = os.path.join(datadir, f"load{load}.pkl")

    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    
    return data

def load_data(args):
    dataname = args.dataset
    load = args.load

    datadict = load_from_pkl(dataname, load)

    num_data_per_class = len(datadict[0])
    if args.num_train + args.num_validation + args.num_test > num_data_per_class:
        raise ValueError(f"Number of data exceeds the range of number of dataset {num_data_per_class}, "
        f"please reduce the number of training, validation, or test data.")

    # split the dataset
    train_datadict = {key: datadict[key][:args.num_train] for key in datadict.keys()}
    val_datadict = {key:datadict[key][args.num_train : args.num_train + args.num_validation] for key in datadict.keys()}
    test_datadict = {key:datadict[key][-args.num_test:] for key in datadict.keys()}

    return train_datadict, val_datadict, test_datadict

def load_big_data():
    dataname = "Bearing"
    loads = [0,1,2,3]

    datadict = load_from_pkl(dataname, 0)
    for load in loads[1:]:
        subdata = load_from_pkl(dataname, load)
        for key in datadict.keys():
            datadict[key] = np.concatenate((datadict[key], subdata[key]), axis=0)
    
    return datadict


def label_data_indices(args):

    num_labeled_per_class = math.ceil(args.num_train * args.label_ratio)
    classes = list(int(i) for i in args.labels.split(","))
    num_class = len(classes)

    indices = np.arange(args.num_train * num_class).reshape(num_class, args.num_train)
    labeled_indices = indices[:, :num_labeled_per_class].reshape(-1,)

    return list(labeled_indices)

def load_tgt_data(args):
    dataname = args.tgt_data
    load = args.tgt_load
    datadict = load_from_pkl(dataname, load)

    num_data_per_class = len(datadict[0])
    if args.num_train + args.num_validation + args.num_test > num_data_per_class:
        raise ValueError(f"Number of data exceeds the range of number of dataset {num_data_per_class}, "
        f"please reduce the number of training, validation, or test data.")

    # split the dataset
    train_datadict = {key: datadict[key][:args.num_train] for key in datadict.keys()}
    val_datadict = {key:datadict[key][args.num_train : args.num_train + args.num_validation] for key in datadict.keys()}
    test_datadict = {key:datadict[key][-args.num_test:] for key in datadict.keys()}

    return train_datadict, val_datadict, test_datadict