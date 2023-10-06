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

# ========== Multi-layer generator model
# ===== Discriminator model size
inputDimD = 200
dimD1 = 256
dimD2 = 512
dimD3 = 1024
dimD4 = 2048
dimD5 = 4096
dimD6 = 2048
dimD7 = 1024
dimD8 = 512
dimD9 = 256
dimD10 = 128
dimD11 = 64
dimD12 = 32
dimD13 = 16
dimD14 = 8
dimD15 = 4
dimD16 = 2
dimD17 = 1


# ===== Torch NN class setup for discriminator
class Discriminator(nn.Module):
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
            nn.Linear(dimD11, dimD12),  # hidden layer 11
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD12, dimD13),  # hidden layer 12
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD13, dimD14),  # hidden layer 13
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD14, dimD15),  # hidden layer 14
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD15, dimD16),  # hidden layer 16
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD16, dimD17),  # output layer
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)




