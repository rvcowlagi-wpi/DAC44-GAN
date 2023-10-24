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
# DIFFERENCE FROM V01
# This version has more hidden layers

device = "cuda"

start_clock = time.time()

n_features = 75
n_discretization = 25
my_batch_size = 64

data_mintime = np.array(pd.read_csv('Data/data_mintime_3_4000.txt'))
train_loader = torch.utils.data.DataLoader(data_mintime, batch_size=my_batch_size)

# Trajectory endpoints; these are fixed in the dataset
x_init = [0.0, 0.8]
x_term = [-0.8, -0.9]

# STEP 3A: Create generator model class
input_dimG = 20
dimG_1 = 64
dimG_2 = 100
dimG_3 = 225
dimG_4 = 400
# dimG_5 = 441
dimG_5 = 625
dimG_6 = 900


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dimG, dimG_1),  # input layer
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_1, dimG_2),  # hidden layer 1
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_2, dimG_3),  # hidden layer 2
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_3, dimG_4),  # hidden layer 3
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_4, dimG_5),  # hidden layer 4
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG_5, dimG_6),  # hidden layer 4
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            # nn.Linear(dimG_6, dimG_7),  # hidden layer 5
            # nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),
            # nn.Linear(1024, 2048),  # hidden layer 6
            # nn.ReLU(),
            nn.Linear(dimG_6, n_features)  # output layer
        )

    def forward(self, x):
        return self.layers(x)
        # return torch.tanh(self.layers(x))


# STEP 3B: Create discriminator model class
dimD_1 = 900
dimD_2 = 625
# dimD_3 = 441
dimD_3 = 400
dimD_4 = 225
dimD_5 = 100
dimD_6 = 25

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_features, dimD_1),  # input layer
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_1, dimD_2),  # hidden layer 1
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_2, dimD_3),  # hidden layer 2
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_3, dimD_4),  # hidden layer 3
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimD_4, dimD_5),  # hidden layer 3
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            # nn.Linear(dimD_5, dimD_6),  # hidden layer 4
            # nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),
            # nn.Linear(64, 32),  # hidden layer 5
            # nn.ReLU(),
            nn.Linear(dimD_5, 1),  # hidden layer 6
            # nn.Sigmoid()  # output layer
        )

    def forward(self, x):
        return self.layers(x)


# STEP 4: Instantiate generator and discriminator classes
def main():
    modelG = Generator()
    modelD = Discriminator()
    modelG.to(device)
    modelD.to(device)
    modelD.train()
    modelG.train()
    # STEP 5: Instantiate loss and optimizer
    criterion = nn.SmoothL1Loss().to(device)  # nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
    # loss_dg = nn.BCELoss().to(device)

    learning_rate_gen = 0.00001
    learning_rate_disc = 0.00001
    optimizerD = torch.optim.RMSprop(modelD.parameters(), lr=learning_rate_disc)
    optimizerG = torch.optim.RMSprop(modelG.parameters(), lr=learning_rate_gen)


    # In a linear layer y = A x + B; our parameters A and B
    # Hence each linear layer has 2 groups of parameters A and B
    # First linear layer size ell_*2 here 2 is input size and ell_
    # is next layers size (here the output)
    # Readout layer size ; the size of out output

    # print('\nGenerator:')
    # print('\t length of model', len(list(modelG.parameters())))
    # print('\t first layer parameter', list(modelG.parameters())[0].size())
    # print('\t first layer bias', list(modelG.parameters())[1].size())
    # print('\t second layer parameter', list(modelG.parameters())[2].size())
    # print('\t second layer bias', list(modelG.parameters())[3].size())
    #
    # print('\nDiscriminator: ')
    # print('\t length of model', len(list(modelD.parameters())))
    # print('\t first layer parameter', list(modelD.parameters())[0].size())
    # print('\t first layer bias', list(modelD.parameters())[1].size())
    # print('\t second layer parameter', list(modelD.parameters())[2].size())
    # print('\t second layer bias', list(modelD.parameters())[3].size())

    # STEP 7: Train the GAN


    # Label for fake and real data"""
    # def ones_target(size):
    #     #   Tensor containing ones, with shape = size    """
    #     data = Variable(torch.ones(size, 1)).to(device)
    #     return data
    #
    #
    # def zeros_target(size):
    #     #   Tensor containing zeros, with shape = size    """
    #     data = Variable(torch.zeros(size, 1)).to(device)
    #     return data


    n_epochs = 2000
    n_iter = 0
    G_losses = []
    D_losses = []
    WEIGHT_CLIP = 0.01
    CRITIC_ITERATIONS = 2
    for epoch in range(n_epochs):
        print('\n======== Epoch ', epoch, ' ========')
        for i, (traj) in enumerate(train_loader):

            for _ in range(CRITIC_ITERATIONS):
                # DISCRIMINATOR
                # Reset gradients
                optimizerD.zero_grad()
                modelD.zero_grad()

                # Train with real examples from dataset
                inputs = traj.float().to(device)

                y_d = modelD(inputs)

                # Calculate errorD and backpropagate
                #errorD_real = loss_dg(y_d, ones_target(len(traj)))
                # errorD_real.backward()

                # Train with fake examples from generator
                z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, input_dimG]).to(device)  # latent vector
                y_g = modelG(z)
                y_d_fake = modelD(y_g)

                # Calculate errorD and backpropagate
                # errorD_fake = loss_dg(y_d_fake, zeros_target(my_batch_size))
                # errorD_fake.backward()
                errorD = -(torch.mean(y_d) - torch.mean(y_d_fake))
                # Compute errorD of D as sum over the fake and the real batches"""
                #errorD = errorD_fake + errorD_real
                errorD.backward()

                # Update D
                optimizerD.step()

                D_losses.append(errorD.item())
                for p in modelD.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)


            # GENERATOR
            # Reset gradients
            modelG.zero_grad()
            optimizerG.zero_grad()

            z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, input_dimG]).to(device)  # latent vector
            y_gG = modelG(z)
            y_d_fakeG = modelD(y_gG)  # Since we just updated D, perform another forward pass of all-fake batch through D

            # Calculate errorD and backpropagate """
            # errorG = loss_dg(y_d_fakeG, ones_target(my_batch_size))
            errorG = -torch.mean(y_d_fakeG)
            errorG.backward()
            # # Update G
            optimizerG.step()
            #
            # loss = criterion(inputs, y_gG)  # Difference between real and fake trajectory points

            n_iter += 1

            # Save Losses for plotting later
            G_losses.append(errorG.item())
            avg_errorG = sum(G_losses) / len(G_losses)
            avg_errorD = sum(D_losses) / len(D_losses)

            # if n_iter == (len(train_loader.dataset) / n_features) * n_epochs:  # i % 5 == 0:
            #     print('Iteration: {}. Loss: {}'.format(n_iter, abs(loss.item())))
            #     Disc_percentage = torch.mean(y_d_fakeG)
            #     print('Discriminator output: this is {} % real'.format(100 - (Disc_percentage.item() * 100)))
        print('Epoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'.format(epoch + 1, n_epochs, avg_errorG,avg_errorD))
    torch.save(modelD.state_dict(), 'Discriminator2model_3000.pth')
    stop_clock = time.time()
    elapsed_hours = int((stop_clock - start_clock) // 3600)
    elapsed_minutes = int((stop_clock - start_clock) // 60 - elapsed_hours * 60)
    elapsed_seconds = (stop_clock - start_clock) - elapsed_hours * 3600 - elapsed_minutes * 60

    print('\nElapsed time ' + str(elapsed_hours) + ':' + str(elapsed_minutes) + ':' + "%.2f" % elapsed_seconds)
    # Save generator outputs
    with open('Data/y_gG.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(my_batch_size):
            row = ['0'] + y_gG[i].tolist()  # add a zero at the beginning of each row
            writer.writerow(row)
    # # STEP 8: Plot a few sample generator results
    t = torch.linspace(0, 10, n_features).to("cpu")
    fig1, ax1 = plt.subplots(3, 3)

    px = 1 / plt.rcParams['figure.dpi']
    fig1.set_figwidth(1800 * px)
    fig1.set_figheight(1600 * px)
    for m1 in range(0, 3):
        for m2 in range(0, 3):
            z = torch.distributions.uniform.Uniform(-1, 1).sample([1, input_dimG]).to(device)
            y = modelG(z).to("cpu")

            y_plt = y[0].detach().numpy()

            # y_plt_x1 = y_plt[(n_discretization):(2*n_discretization)]
            # y_plt_x2 = y_plt[(2*n_discretization):(3*n_discretization)]

            y_plt_x1 = y_plt[0:n_discretization]
            y_plt_x2 = y_plt[n_discretization:(2 * n_discretization)]

            ax1[m1, m2].plot(y_plt_x1, y_plt_x2)
            ax1[m1, m2].plot(x_init[0], x_init[1], marker="o", markersize=5)
            ax1[m1, m2].plot(x_term[0], x_term[1], marker="s", markersize=5)
            ax1[m1, m2].axis('equal')

            # y_plt_x1 = data_mintime[(3 * m1 + m2), 0:n_discretization]
            # y_plt_x2 = data_mintime[(3 * m1 + m2), n_discretization:(2 * n_discretization)]
            #
            # ax1[m1, m2].plot(y_plt_x1, y_plt_x2)
            # ax1[m1, m2].plot(x_init[0], x_init[1], marker="o", markersize=5)
            # ax1[m1, m2].plot(x_term[0], x_term[1], marker="s", markersize=5)
            # ax1[m1, m2].axis('equal')

    # figname_ = "Results/2023-01-14-LTI/lti_1d_ep" + str(n_epochs) + "_ex" + \
    #            str(len(train_loader.dataset) // n_features) + "_pt" + str(n_features) + ".png"
    # figname_ = "Results/2023-01-14-LTI/lti_2d_ep" + str(n_epochs) + "_ex" + \
    #            str(len(train_loader.dataset)) + "_pt" + str(n_features) + ".png"
    # fig1.savefig(figname_, bbox_inches='tight')


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
#
