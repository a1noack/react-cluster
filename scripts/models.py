"""This module holds the classes for Siamese networks and the classes for the
datasets needed to train and evaluate them"""
import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class SiameseNet(nn.Module):
    """A Siamese network with a variable number of layers
    that uses a weighted L1 distance metric during training"""
    def __init__(self, in_size, hid_size, out_size, n_layer=4, drop_prob=.5, group_size=1):
        super(SiameseNet, self).__init__()
        self.hidden = nn.ModuleList()
        for i in range(n_layer):
            n_in = n_out = hid_size
            if i == 0:
                n_in = in_size
            elif i == n_layer - 1:
                n_out = out_size
            self.hidden.append(nn.Linear(n_in, n_out))
        self.out = nn.Linear(out_size, 1)
        self.drop_prob = drop_prob
        self.n_layer = n_layer
        self.group_size = group_size
        self.out_size = out_size

    def forward_one(self, x):
        # embed x by passing it through the network
        for i, l in enumerate(self.hidden):
            x = l(x)
            if i < self.n_layer - 1:
                x = F.dropout(F.relu(x), p=self.drop_prob)  # last layer doesn't need activation or dropout
        # x = F.normalize(x, p=2, dim=-1)  # normalize magnitude of embedded vectors in batch; turn on if using cos sim
        return x

    # use this with BinaryCrossEntropy with Logits loss
    def forward(self, x1, x2, group_size=None):
        group_size = self.group_size if group_size is None else group_size
        # do not use group loss
        if group_size == 1:
            out1 = self.forward_one(x1)
            out2 = self.forward_one(x2)

        # use group loss (assumes x1 and x2 have shape (batch_size, group_size, input_size)
        else:
            # pass through network and reshape
            out1 = self.forward_one(x1).reshape(-1, group_size, self.out_size)
            out2 = self.forward_one(x2).reshape(-1, group_size, self.out_size)

            # average across groups
            out1 = out1.mean(dim=1)  # shape should now be (batch_size, out_size)
            out2 = out2.mean(dim=1)

        # get the distance between the samples in the embedding space
        dist = torch.abs(out1 - out2)
        out = self.out(dist)
        out = out if group_size > 1 else out.squeeze()

        return out


class SiameseDataset(Dataset):
    """Define a Dataset where each instance is a pair of attacked samples — or a pair of groups
    of attacked samples if the group_size parameter is greater than one – and a label;
    the label indicates whether the attacked samples were created by the same
    attack method (label=1) or by different attack methods (label=0)"""
    def __init__(self, df, freq=None, group_size=1):
        self.labels = df['label'].values
        self.data = df.drop(['label', 'key'], axis=1, errors='ignore')
        self.freq = 1 / len(set(self.labels)) if freq is None else freq  # set 'same' class frequency
        self.group_size = group_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx1):
        # get a pair of attacks
        if self.group_size == 1:
            # get sample at provided index
            x1, y1 = self.data.iloc[idx1].values, self.labels[idx1]

            # get an index of a sample produced by the same attack method
            if np.random.randn() < self.freq:
                idx2 = np.random.choice(np.where(self.labels == y1)[0])
                y = 1

            # get an index of a sample produced by a different attack method
            else:
                idx2 = np.random.choice(np.where(self.labels != y1)[0])
                y = 0

            # get sample at this second index
            x2 = self.data.iloc[idx2].values

        # create two groups where each group's attacks were all created by the same attack method
        else:
            # get a group of samples with the label at the given index
            y1 = self.labels[idx1]
            idxs1 = np.random.choice(np.where(self.labels == y1)[0], size=self.group_size, replace=False)
            x1 = self.data.iloc[idxs1].values

            # get indexes for a group of samples produced by the same attack method that produced first group
            if np.random.randn() < self.freq:
                available_idxs = np.array(sorted(set(np.where(self.labels == y1)[0]) - set(idxs1)))
                idxs2 = np.random.choice(available_idxs, size=self.group_size, replace=False)
                y = np.array([1], dtype=np.float32)

            # get indexes for a group of samples produced by a different attack method
            else:
                y2 = y1
                while y2 == y1:  # randomly choose label for second group of samples
                    y2 = np.random.choice(self.labels)
                idxs2 = np.random.choice(np.where(self.labels == y2)[0], size=self.group_size, replace=False)
                y = np.array([0], dtype=np.float32)

            # get group of samples at second group of indices
            x2 = self.data.iloc[idxs2].values

            # check to make sure that the two groups have the same, correct size
            assert x1.shape[0] == self.group_size == x2.shape[0], "Incorrect group size!"

        return x1, x2, y


class NormalDataset(Dataset):
    """Returns a single sample and a label or a group of samples that
    were all produced by the same attack method and their label"""

    def __init__(self, df, group_size=1, reuse_samples=False, return_keys=False):
        self.labels = df['label'].values
        self.keys = df['key'].values
        self.data = df.drop(['label', 'key'], axis=1, errors='ignore')
        self.group_size = group_size
        self.reuse_samples = reuse_samples
        self.used_idxs = []
        self.return_keys = return_keys

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Returns x, the features for a single sample or a group of samples, y, the label for the
        sample or group of samples, and keys, the key for the sample or set of keys for the samples
        in the group"""

        # if not using groups of samples, just return one sample
        if self.group_size == 1:
            x, y = self.data.iloc[idx].values, self.labels[idx]  # y is a string; e.g. 'hotflip'
            keys = [self.keys[idx]]

        # return a group of self.group_size samples
        else:
            y = random.choice(self.labels)  # y is a string; e.g. 'hotflip'

            # each sample can appear in MULTIPLE groups
            if self.reuse_samples:
                idxs1 = np.random.choice(np.where(self.labels == y)[0], size=self.group_size, replace=False)

            # each sample can only appear in ONE group
            else:
                try:
                    idxs1 = np.random.choice(sorted(set(np.where(self.labels == y)[0]) - set(self.used_idxs)),
                                             size=self.group_size, replace=False)
                    self.used_idxs.extend(idxs1)
                except ValueError:
                    raise StopIteration  # stop iterating if all samples have been used up

            keys = [self.keys[idx] for idx in idxs1]
            x = self.data.iloc[idxs1].values

        if self.return_keys:
            return x, y, keys
        else:
            return x, y
