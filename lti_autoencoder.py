# ========== Dependencies
import torch
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time
from datetime import datetime
import model_setup_encoder
import model_setup_G_rulesD

# ===== Initialize
device = "cpu"
start_clock = time.time()

nFeatures = model_setup_G_rulesD.nFeatures
my_batch_size = 1

# ===== Instantiate encoder, loss function, and optimizer
model_encoder = model_setup_encoder.Autoencoder()
model_encoder.to(device)
lossFcn_en = nn.MSELoss().to(device)
learningRateG = 1E-4
optimizer_en = torch.optim.Adam(model_encoder.parameters(), lr=learningRateG)

# ===== Instantiate generator, loss function, and optimizer
modelG = model_setup_G_rulesD.Generator()
modelG.to(device)
lossFcnG = nn.MSELoss().to(device)
learningRateG = 1E-4
optimizerG = torch.optim.Adam(modelG.parameters(), lr=learningRateG)

# ===== Initialize training
nEpochs = 5
nIter = 0
lossRecordEn = []

# Label for fake and real data"""
def ones_target(size):
    #   Tensor containing ones, with shape = size    """
    data = Variable(torch.ones(size, 1)).to(device)
    return data


def zeros_target(size):
    #   Tensor containing zeros, with shape = size    """
    data = Variable(torch.zeros(size, 1)).to(device)
    return data


# ===== Train
for epoch in range(nEpochs):
    print('\n======== Epoch ', epoch, ' ========')
    for i, (traj) in enumerate(model_setup_encoder.train_loader):

        modelG.zero_grad()
        model_encoder.zero_grad()

        # input real examples from dataset
        inputs = traj.float()  # .to(device)

        # ----- Pass real data through encoder
        y_en = model_encoder(inputs)

        # ----- Pass encoder output through Generator(generator acts as decoder)
        y_gG = modelG(y_en).to(device)

        # Difference between generated and real data
        train_en = inputs - y_gG

        # ----- Find gradient and update D
        loss_autoencoder = lossFcn_en(torch.transpose(train_en, 0, 1), zeros_target(200))
        loss_autoencoder.backward()
        optimizer_en.step()
        optimizerG.step()

        # ----- Book-keeping
        nIter += 1

        # Save Losses for plotting later
        lossRecordEn.append(loss_autoencoder.item())

# ===== Save the trained autoencoder model (Generator which is the decoder)
torch.save(modelG.state_dict(), 'lti_autoencoder.pt')

stopClock = time.time()
elapsedHours = int((stopClock - start_clock) // 3600)
elapsedMinutes = int((stopClock - start_clock) // 60 - elapsedHours * 60)
elapsedSeconds = (stopClock - start_clock) - elapsedHours * 3600 - elapsedMinutes * 60

print('\nElapsed time ' + str(elapsedHours) + ':' + str(elapsedMinutes) + ':' + "%.2f" % elapsedSeconds)

# Plot loss
fig1 = plt.figure()
plt.plot(lossRecordEn)
plt.title("Loss during autoencoder training")
plt.xlabel("Iteration")
plt.ylabel("Loss")


# Plot a few sample generator results
n_plot_row = 5
n_plot_col = 2
# fig3 = plt.figure()
fig, ax = plt.subplots(n_plot_row, n_plot_col)

# --- For 2 state
t = torch.linspace(0, 10, model_setup_G_rulesD.nTimeStamps).to("cpu")
px = 1/plt.rcParams['figure.dpi']
fig.set_figwidth(1800*px)
fig.set_figheight(1500*px)
modelG.to("cpu")
for m1 in range(0, n_plot_row):
    for m2 in range(0, n_plot_col):
        z = torch.randn([1, model_setup_G_rulesD.inputDimG])
        y = modelG(z)
        y = y.reshape([nFeatures, 1])
        x1_plt = y[0:model_setup_G_rulesD.nTimeStamps].detach().numpy()
        # x1_plt = inputs[0][0:model_setup_G_rulesD.nTimeStamps].detach().numpy()
        x2_plt = y[model_setup_G_rulesD.nTimeStamps:nFeatures].detach().numpy()
        ax[m1, m2].plot(t, x1_plt)
        ax[m1, m2].plot(t, x2_plt)

plt.show()
