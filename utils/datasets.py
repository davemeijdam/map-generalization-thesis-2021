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

labels_map = {
    0: "No Change",
    1: "Douglas-Peucker 0.5",
    2: "Douglas-Peucker 0.1",
    3: "Douglas-Peucker 0.05",
    4: "Douglas-Peucker 0.01",
    5: "Douglas-Peucker 0.005",
    6: "Douglas-Peucker 0.001",
    7: "Visvalingam-Whyatt 0.5",
    8: "Visvalingam-Whyatt 0.1",
    9: "Visvalingam-Whyatt 0.05",
    10: "Visvalingam-Whyatt 0.01",
    11: "Visvalingam-Whyatt 0.005",
    12: "Visvalingam-Whyatt 0.001",
    13: "Visvalingam-Whyatt 0.0005",
    14: "Visvalingam-Whyatt 0.0001",
    15: "Visvalingam-Whyatt 0.00005"
}

class GeoSpatialDataset():

    def __init__(self,labels, data):
        self.labels = pickle.load(open(str(datadir + 'y_train.p')))
        self.data = pickle.load(open(str(datadir + 'y_train.p')))

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