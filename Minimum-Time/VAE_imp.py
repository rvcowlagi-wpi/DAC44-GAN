#This code implements the S-VAE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import time
import csv
device = "cuda"
# Hyperparameters
hidden_size_E1 = 324
hidden_size_E2 = 225
hidden_size_E3 = 196
hidden_size_E4 = 125
hidden_size_E5 = 100
hidden_size_E6 = 81

hidden_size_D1 = 81
hidden_size_D2 = 100
hidden_size_D3 = 125
hidden_size_D4 = 196
hidden_size_D5 = 225
hidden_size_D6 = 324
latent_size = 32
learning_rate = 1e-3
epochs = 700
n_features = 400
n_discretization = 25
my_batch_size = 32
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_size_E1),
            nn.ReLU(),
            nn.Linear(hidden_size_E1, hidden_size_E2),
            nn.ReLU(),
            nn.Linear(hidden_size_E2, hidden_size_E3),
            nn.ReLU(),
            nn.Linear(hidden_size_E3, hidden_size_E4),
            nn.ReLU(),
            nn.Linear(hidden_size_E4, hidden_size_E5),
            nn.ReLU(),
            nn.Linear(hidden_size_E5, hidden_size_E6),
            nn.ReLU(),
            nn.Linear(hidden_size_E6, latent_size * 2)  # *2 for mean and variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size_D1),
            nn.ReLU(),
            nn.Linear(hidden_size_D1, hidden_size_D2),
            nn.ReLU(),
            nn.Linear(hidden_size_D2, hidden_size_D3),
            nn.ReLU(),
            nn.Linear(hidden_size_D3, hidden_size_D4),
            nn.ReLU(),
            nn.Linear(hidden_size_D4, hidden_size_D5),
            nn.ReLU(),
            nn.Linear(hidden_size_D5, hidden_size_D6),
            nn.ReLU(),
            nn.Linear(hidden_size_D6, n_features),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x):
        # Encode
        latent_params = self.encoder(x)
        mu, logvar = torch.chunk(latent_params, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def loss_function(x_recon, x, mu, logvar):
    # Reconstruction loss (binary cross-entropy)
    recon_loss = nn.MSELoss(reduction='sum')(x_recon, x)

    # KL divergence regularization
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
    # Set random seed for reproducibility
    torch.manual_seed(42)


    # LEAKY_RELU = 0.01
    # Create VAE model


def main():
    vae = VAE().to(device)
    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    data_mintime = np.array(pd.read_csv('Data/data_mintime_7_4000_grid.txt'))
    train_loader = torch.utils.data.DataLoader(data_mintime, batch_size=my_batch_size)
    vae.train()
    best_epoch = -1
    best_loss = float('inf')
    # Training loop
    for epoch in range(epochs):
        for i, (traj) in enumerate(train_loader):
            inputs = traj.float().to(device)
            # inputs = inputs[:,50: 125]
            # print(inputs.shape)
            # Forward pass
            x_recon, mu, logvar = vae(inputs)

            # Compute loss
            loss = loss_function(x_recon, inputs, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                # Save the model for the best epoch
                best_vae_path = 'best_vae_model.pth'
                torch.save(vae.state_dict(), best_vae_path)
    print(f"Best epoch: {best_epoch + 1}, Best loss: {best_loss}")

    # Generate samples from the learned VAE
    vae.eval()
    all_reconstructed_samples = []

    with torch.no_grad():
        for i, (traj) in enumerate(train_loader):
            inputs = traj.float().to(device)
            # inputs = inputs[:,50:125]
            x_recon, _, _ = vae(inputs)
            all_reconstructed_samples.append(x_recon.cpu().numpy())

    # Extract the last set of reconstructed samples
    last_reconstructed_samples = all_reconstructed_samples[-2]

    # Save the last set of reconstructed samples to a CSV file
    reconstructed_csv_path = 'Data/last_reconstructed_samples_4000_grid.csv'
    np.savetxt(reconstructed_csv_path, last_reconstructed_samples, delimiter=',')
    # Save the entire VAE model
    vae_path = 'vae_model.pth'
    torch.save(vae.state_dict(), vae_path)

    # # STEP 8: Plot a few sample generator results
    x_init = [0.0, 0.8]
    x_term = [-0.8, -0.9]
    t = torch.linspace(0, 10, n_features).to("cpu")
    fig1, ax1 = plt.subplots(3, 3)

    px = 1 / plt.rcParams['figure.dpi']
    fig1.set_figwidth(1800 * px)
    fig1.set_figheight(1600 * px)
    for m1 in range(0, 3):
        for m2 in range(0, 3):
            z_sample = torch.randn(1, latent_size).to(device)
            y = vae.decoder(z_sample).detach().cpu().numpy()
            y_plt = y[0]
            y_plt_x1 = y_plt[0:n_discretization]
            y_plt_x2 = y_plt[n_discretization:(2 * n_discretization)]
            ax1[m1, m2].plot(y_plt_x1, y_plt_x2)
            ax1[m1, m2].plot(x_init[0], x_init[1], marker="o", markersize=5)
            ax1[m1, m2].plot(x_term[0], x_term[1], marker="s", markersize=5)
            ax1[m1, m2].axis('equal')

    plt.show()









if __name__ == "__main__":
    main()