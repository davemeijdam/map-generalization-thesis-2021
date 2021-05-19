import pandas as pd
import numpy as np
import torch
import pickle
import warnings
from time import time
import operator
from collections import Counter
import json
import os
import math
import random

datadir = '/Users/davemeijdam/Documents/Data Science/Master/Master Thesis/Data/'

resources = [
    'X_train.p',
    'X_test.p',
    'y_train.p',
    'y_test.p'
]

class GeoSpatialDataset():

    def __init__(self,labels, data, train=True):
        if train == True:
            self.labels = pickle.load(open(str(datadir + 'y_train.p')))
            self.data = pickle.load(open(str(datadir + 'y_train.p')))

        if train == False:
            self.labels = pickle.load(open(str(datadir + 'y_test.p')))
            self.data = pickle.load(open(str(datadir + 'y_test.p')))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data