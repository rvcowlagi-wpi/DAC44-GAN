# PROGRAM DESCRIPTION: This code loads the trained Generator model

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
from HGAN_mintime import Generator
from sklearn. metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelG = Generator()
    modelG.to(device)
    modelG.load_state_dict(torch.load('Generator_model_7.pth'))
    modelG.eval()

# Trajectory parameters
    x_init = [0.0, 0.8]
    x_term = [-0.8, -0.9]
    n_features = 175
    n_discretization = 25
    my_batch_size = 128

# Prepare the test dataset
    input_dimG = 20
    z = torch.distributions.uniform.Uniform(-1, 1).sample([my_batch_size, input_dimG]).to(device)
    ham,y_pred = modelG(z)

    for output in ham:
        print(output)

# Save generator outputs
    with open('Data/y_pred.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(my_batch_size):
            row = ['0'] + y_pred[i].tolist()  # add a zero at the beginning of each row
            writer.writerow(row)

#Plot a few sample generator results
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
    plt.show()
if __name__ == "__main__":
    main()