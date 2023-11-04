import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time

# ===== Problem size
nFeatures = 200
nTimeStamps = 100
dt_ = 0.1
my_batch_size = 1

# ======== Load the real training data
data_2d = np.array(pd.read_csv('Data/data_2d.txt'))
train_loader = torch.utils.data.DataLoader(data_2d, batch_size=my_batch_size)

# ========== Encoder model
inputDimD = 200
dimD1 = 256
dimD2 = 256
dimD3 = 512
dimD4 = 1024
dimD5 = 2048
dimD6 = 4096
dimD7 = 2048
dimD8 = 1024
dimD9 = 512
dimD10 = 256
dimD11 = 100

# ===== Setup of Encoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(inputDimD, dimD1),  # input layer
            nn.LeakyReLU(0.2),
            nn.Linear(dimD1, dimD2),  # hidden layer 1
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD2, dimD3),  # hidden layer 2
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD3, dimD4),  # hidden layer 3
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD4, dimD5),  # hidden layer 4
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD5, dimD6),  # hidden layer 5
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD6, dimD7),  # hidden layer 6
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD7, dimD8),  # hidden layer 7
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD8, dimD9),  # hidden layer 8
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD9, dimD10),  # hidden layer 9
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD10, dimD11),  # hidden layer 10
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD11, 100),  # output layer
        )

    def forward(self, x):
        return self.layers(x)




