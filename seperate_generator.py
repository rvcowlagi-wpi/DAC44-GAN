# PROGRAM DESCRIPTION: This code separately trains the generator on the zero Hamiltonian criterion

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.autograd.variable import Variable
import time
import csv


device = "cuda"
start_clock = time.time()

n_features = 175
n_discretization = 25
my_batch_size = 32

# Trajectory parameters
V = 0.05

# STEP 1: Create generator model class with the Hamiltonian function at the output
input_dimG = 20
HIDDENSIZE = 32
NEG_SLOPE = 0.01
DROPOUT = 0.2

# Generator architecture that works the best
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dimG, HIDDENSIZE),
            nn.LeakyReLU(NEG_SLOPE),  # Use LeakyReLU activation with a negative slope of NEG_SLOPE
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDENSIZE, HIDDENSIZE),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDENSIZE, HIDDENSIZE),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDENSIZE, HIDDENSIZE),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDENSIZE, HIDDENSIZE),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDENSIZE, HIDDENSIZE),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDENSIZE, HIDDENSIZE),
            nn.LeakyReLU(NEG_SLOPE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDENSIZE, n_features)
        )
# #
    def calculateHamiltonian(self, arr):
        H = list()
        for state in range(n_discretization):
            hamil = 1 + arr[:, state + 75] * V * torch.cos(arr[:,state + 50]) + arr[:, state + 75] * arr[:, state + 125] + arr[:, state + 100] * V * \
                torch.sin(arr[:, state + 50]) + arr[:, state + 100] * arr[:, state + 150]
            H.append(hamil)
        H = torch.stack(H)
        return torch.transpose(H,0,1)

    def forward(self, x):
        model_output = self.layers(x)
        hamil = self.calculateHamiltonian(model_output)
        return hamil,model_output

def main():
    # STEP 2: Instantiate generator and discriminator classes
    modelG = Generator()
    modelG.to(device)

    # STEP 3: Instantiate loss and optimizer
    loss_dg = nn.MSELoss().to(device)
    learning_rate_gen = 0.01
    optimizerG = torch.optim.Adam(modelG.parameters(), lr=learning_rate_gen)

    # STEP 4: Train the GAN
    def ones_target(size):
        #   Tensor containing ones, with shape = size    """
        data = Variable(torch.ones(size, 1)).to(device)
        return data

    def zeros_target(size):
        #   Tensor containing zeros, with shape = size    """
        data = Variable(torch.zeros(size, 1)).to(device)
        return data

    n_epochs = 4000
    n_iter = 0
    G_losses = []

    for epoch in range(n_epochs):

        print('\n======== Epoch ', epoch, ' ========')
        # Reset gradients of Generator
        modelG.zero_grad()
        optimizerG.zero_grad()
        z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, input_dimG]).to(device)  # latent vector
        hamil, y_pred = modelG(z)

        # Calculate errorD and backpropagate """
        loss_MSE = loss_dg(hamil, torch.zeros_like(hamil))
        errorG = loss_MSE
        errorG.backward()

        # Update G
        optimizerG.step()
        n_iter += 1
        G_losses.append(errorG.item())
        avg_errorG = sum(G_losses) / len(G_losses)
        print('Epoch [{}/{}], Generator Loss: {:.4f}'.format(epoch + 1, n_epochs, avg_errorG))

# STEP 5: Saves the Generator model
    torch.save(modelG.state_dict(), 'Generator_model_mse.pth')

# STEP 6: Saves the last output from the generator in a csv file
    with open('Data/y_pred.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(my_batch_size):
                    row = ['0'] + y_pred[i].tolist()  # add a zero at the beginning of each row
                    writer.writerow(row)
    stop_clock = time.time()
    elapsed_hours = int((stop_clock - start_clock) // 3600)
    elapsed_minutes = int((stop_clock - start_clock) // 60 - elapsed_hours * 60)
    elapsed_seconds = (stop_clock - start_clock) - elapsed_hours * 3600 - elapsed_minutes * 60

    print('\nElapsed time ' + str(elapsed_hours) + ':' + str(elapsed_minutes) + ':' + "%.2f" % elapsed_seconds)

# STEP 7: Calculates Hamiltonian of a few sample generator results
    for m1 in range(0, 3):
        for m2 in range(0, 3):
            z = torch.distributions.uniform.Uniform(-1, 1).sample([1, input_dimG]).to(device)
            _, y = modelG(z)
            y = y.to("cpu")

            # Calculate Hamiltonian for the plotted trajectory
            hamil = modelG.calculateHamiltonian(y)
            print('Hamiltonian for the plotted trajectory:', hamil)


if __name__ == "__main__":
    main()
