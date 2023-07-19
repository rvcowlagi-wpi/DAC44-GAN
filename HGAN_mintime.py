# PROGRAM DESCRIPTION: This code trains the GAN model implementing the Hamiltonian criterion

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time
import csv


device = "cuda"
start_clock = time.time()

n_features = 175
n_discretization = 25
my_batch_size = 32

# STEP 1: Loading the data
data_mintime = np.array(pd.read_csv('Data/data_mintime_2.txt'))
train_loader = torch.utils.data.DataLoader(data_mintime, batch_size=my_batch_size)

# Trajectory parameters; these are fixed in the dataset
x_init = [0.0, 0.8]
x_term = [-0.8, -0.9]
V = 0.05

# STEP 2: Create Discriminator model class
input_dimG = 20
dimD_1 = 900
dimD_2 = 625
dimD_3 = 484
dimD_4 = 324
dimD_5 = 125
dimD_6 = 25

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(50, dimD_1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_1, dimD_2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_2, dimD_3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_3, dimD_4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_4, dimD_5),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_5, dimD_6),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

#STEP 3: Create generator class with the Hamiltonian at the output
dimG_1 = 64
dimG_2 = 100
dimG_3 = 225
dimG_4 = 400
dimG_5 = 400
dimG_6 = 256

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dimG, dimG_1),
            nn.LeakyReLU(0.1),  # Use LeakyReLU activation with a negative slope of 0.2
            nn.Dropout(0.2),
            nn.Linear(dimG_1, dimG_2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(dimG_2, dimG_3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(dimG_3, dimG_4),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(dimG_4, dimG_5),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(dimG_5, dimG_6),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(dimG_6, n_features)
        )

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

# STEP 4: Instantiate generator and discriminator classes
def main():
    modelG = Generator()
    modelD = Discriminator()
    modelD.to(device)
    modelG.to(device)

# STEP 5: Instantiate loss and optimizer
    loss_dg = nn.MSELoss().to(device)
    learning_rate_gen = 0.01
    learning_rate_disc = 0.01
    optimizerD = torch.optim.SGD(modelD.parameters(), lr=learning_rate_disc)
    optimizerG = torch.optim.SGD(modelG.parameters(), lr=learning_rate_gen)

# STEP 6: Train the GAN
    def ones_target(size):
        #   Tensor containing ones, with shape = size    """
        data = Variable(torch.ones(size, 1)).to(device)
        return data

    def zeros_target(size):
        #   Tensor containing zeros, with shape = size    """
        data = Variable(torch.zeros(size, 1)).to(device)
        return data

    n_epochs = 3
    n_iter = 0
    G_losses = [] # Total generator loss
    GD_losses = [] # loss from Discriminator
    D_losses = [] # Discriminator loss
    MSE_losses = [] # Hamiltonian loss

    for epoch in range(n_epochs):
        for i, (traj) in enumerate(train_loader):
            print('\n======== Epoch ', epoch, ' ========')
            optimizerD.zero_grad()
            modelD.zero_grad()

            # Train with real examples from dataset
            inputs = traj.float().to(device)
            y_d = modelD(inputs)

            # Calculate errorD and backpropagate
            errorD_real = loss_dg(y_d, ones_target(len(y_d)))

            # Train with fake examples from generator
            z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, input_dimG]).to(device)  # latent vector
            _,y = modelG(z)
            yd = y[:, 0:50]
            y_d_fake = modelD(yd)

            # Calculate errorD and backpropagate
            errorD_fake = loss_dg(y_d_fake, zeros_target(len(y_d_fake)))

            # Compute errorD of D as sum over the fake and the real batches"""
            errorD = errorD_fake + errorD_real
            errorD.backward()

            # Update D
            optimizerD.step()
            D_losses.append(errorD.item())

            # Reset gradients of Generator
            modelG.zero_grad()
            optimizerG.zero_grad()
            z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, input_dimG]).to(device)  # latent vector
            hamil, y_pred = modelG(z)
            yg = y_pred[:, 0:50]
            y_d_fakeG = modelD(yg)  # Since we just updated D, perform another forward pass of all-fake batch through D

            # Calculate errorD and backpropagate """
            errorG1 = loss_dg(y_d_fakeG, ones_target(my_batch_size))
            loss_MSE = loss_dg(hamil, torch.zeros_like(hamil))
            errorG = loss_MSE + errorG1
            errorG.backward()

            # Update G
            optimizerG.step()
            n_iter += 1
            G_losses.append(errorG.item())
            avg_errorG = sum(G_losses) / len(G_losses)
            GD_losses.append(errorG1.item())
            avg_errorGD = sum(GD_losses) / len(GD_losses)
            MSE_losses.append(loss_MSE.item())
            avg_errorMSE = sum(MSE_losses) / len(MSE_losses)
            D_losses.append(errorD.item())
            avg_errorD = sum(D_losses) / len(D_losses)
            print('Epoch [{}/{}], Generator Loss: {:.4f}'.format(epoch + 1, n_epochs, avg_errorG))
            print('Epoch [{}/{}], Discriminator Loss: {:.4f}'.format(epoch + 1, n_epochs, avg_errorD))
            print('Epoch [{}/{}], GD Loss: {:.4f}'.format(epoch + 1, n_epochs, avg_errorGD))
            print('Epoch [{}/{}], MSE Loss: {:.4f}'.format(epoch + 1, n_epochs, avg_errorMSE))

    # torch.save(modelG.state_dict(), 'Generator_model_7_1.pth')

# STEP 7: Save the last generator output as CSV file
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

# STEP 8: Plot a few sample generator results
    t = torch.linspace(0, 10, n_features).to("cpu")
    fig1, ax1 = plt.subplots(3, 3)
    px = 1 / plt.rcParams['figure.dpi']
    fig1.set_figwidth(1800 * px)
    fig1.set_figheight(1600 * px)
    for m1 in range(0, 3):
        for m2 in range(0, 3):
            z = torch.distributions.uniform.Uniform(-1, 1).sample([1, input_dimG]).to(device)
            _, y = modelG(z)
            y = y.to("cpu")
            # Calculate Hamiltonian for the plotted trajectory
            hamil = modelG.calculateHamiltonian(y)
            y_plt = y[0].detach().numpy()
            y_plt_x1 = y_plt[0:n_discretization]
            y_plt_x2 = y_plt[n_discretization:(2 * n_discretization)]
            ax1[m1, m2].plot(y_plt_x1, y_plt_x2)
            ax1[m1, m2].plot(x_init[0], x_init[1], marker="o", markersize=5)
            ax1[m1, m2].plot(x_term[0], x_term[1], marker="s", markersize=5)
            ax1[m1, m2].axis('equal')
            # Print the Hamiltonian for the plotted trajectory
            print('Hamiltonian for the plotted trajectory:', hamil)
    fig2 = plt.figure()
    fig2.set_figwidth(1200 * px)
    fig2.set_figheight(800 * px)
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
