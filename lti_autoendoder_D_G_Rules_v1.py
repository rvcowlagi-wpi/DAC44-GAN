# ========== Dependencies
import torch
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time
from datetime import datetime
import model_setup_G_rulesD
import model_setup_nnD

# ===== Initialize
device = "cpu"
start_clock = time.time()

nFeatures = model_setup_G_rulesD.nFeatures
my_batch_size = 1

# ===== Load autoencoder as generator
modelG = model_setup_G_rulesD.Generator()
modelG.load_state_dict(torch.load('lti_autoencoder.pt'))
lossFcnG = nn.MSELoss().to(device)
learningRateG = 1E-4
optimizerG = torch.optim.Adam(modelG.parameters(), lr=learningRateG)


# ===== Instantiate discriminator, loss function, and optimizer
modelD = model_setup_nnD.Discriminator()
modelD.to(device)
lossFcnD = nn.BCELoss().to(device)
learningRateG = 1E-4
optimizerD = torch.optim.Adam(modelD.parameters(), lr=learningRateG)


# ===== Initialize training
nEpochs = 5
nIter = 0
lossRecordD = []
lossRecordG = []


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
    for i, (traj) in enumerate(model_setup_nnD.train_loader):

        # DISCRIMINATOR
        # Reset gradients
        modelD.zero_grad()

        # input real examples from dataset
        inputs = traj.float()  # .to(device)

        # ----- Pass real data through D
        y_d = modelD(inputs)

        # Calculate loss in nn D with real data
        loss_real = lossFcnD(y_d, ones_target(len(traj)))

        # ===== GENERATOR =====
        # ----- Reset gradients
        modelG.zero_grad()

        # ----- Sample from latent vector space
        z = torch.randn([my_batch_size, model_setup_G_rulesD.inputDimG]).to(device)

        # ----- Find rules discriminator score (ideal is zero) and loss
        # y_gG1 = y_gG.reshape([nFeatures, 1])
        # rulesScore = model_setup_G_rulesD.rules_d_state1(y_gG1, device)
        # lossGFromRules = lossFcnG(rulesScore, Variable(torch.zeros(nFeatures, 1)).to(device))

        # ----- Pass latent vector through updated G
        y_gG = modelG(z).to(device)

        # ----- Pass fake data through nn D
        y_d_fake = modelD(y_gG)

        # Calculate loss in nn D with fake data
        loss_fake = lossFcnD(y_d_fake, zeros_target(len(traj)))

        # ----- Find gradient and update D
        lossD = loss_real+loss_fake
        # lossD.requires_grad = True
        lossD.backward()
        optimizerD.step()

        # Calculate loss in G from nn D
        y_gG1 = modelG(z).to(device)
        y_d_forG = modelD(y_gG1)
        rulesScore = model_setup_G_rulesD.rules_d_state1(y_gG1.reshape([nFeatures, 1]), device)
        lossGFromRules = lossFcnG(rulesScore, Variable(torch.zeros(nFeatures, 1)).to(device))
        lossGFromnnD = lossFcnD(y_d_forG, ones_target(my_batch_size))

        # ----- Find gradient and update G
        lossG = 0.5 * lossGFromnnD + 0.5 * lossGFromRules
        # lossG.requires_grad = True
        lossG.backward()
        optimizerG.step()

        # ----- Book-keeping
        nIter += 1

        # Save Losses for plotting later
        lossRecordD.append(lossD.item())
        lossRecordG.append(lossG.item())


# # ===== Save the trained model
# torch.save(model_D.state_dict(), 'simple_dynamics_discriminator_nn.pt')

stopClock = time.time()
elapsedHours = int((stopClock - start_clock) // 3600)
elapsedMinutes = int((stopClock - start_clock) // 60 - elapsedHours * 60)
elapsedSeconds = (stopClock - start_clock) - elapsedHours * 3600 - elapsedMinutes * 60

print('\nElapsed time ' + str(elapsedHours) + ':' + str(elapsedMinutes) + ':' + "%.2f" % elapsedSeconds)

# Plot discriminator loss
fig1 = plt.figure()
plt.plot(lossRecordD)
plt.title("Discriminator loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")

# Plot generator loss
fig2 = plt.figure()
plt.plot(lossRecordG)
plt.title("Generator loss during training")
plt.xlabel("Iteration")
plt.ylabel("Generator Loss")


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
