# Script for testing a generator model.
# Takes several sample G outputs and compares against a library
# using L2 distance as a similarity test.
# Also checks for diversity of samples, i.e., whether the distance between samples is zeros.


# ========== Dependencies
import torch
import numpy as np
import model_setup_G_rulesD
import matplotlib.pyplot as plt


# =========== L2 distance calculator
def ell2_distance(z1_, z2_):
    # z1 and z2 are two vectors of equal size
    distance_ = np.sum(np.square(z1_ - z2_))
    return distance_


# # =========== Lyapunov equation
# def lyapunov_fun(x1):
#     # k = 0.5
#     t = torch.arange(1, 10.1, 0.1)
#     p = torch.randn(100, 100)
#     p = torch.mm(p, p.t())
#     x1 = torch.from_numpy(x1)
#     x1 = torch.reshape(x1, (100, 1))
#     x1_transpose = torch.transpose(x1, 0, 1)
#     lyap_1 = torch.matmul(x1_transpose, p)
#     lyap_ = torch.matmul(lyap_1, x1)
#     lyap_.requires_grad = True
#     lyap_.backward()
#     print(t.grad)
#     eig_val, eig_vec = torch.linalg.eig(lyap_)
#
#     if eig_val.real > 0:
#         v_satisfied = 1
#
#     return v_satisfied


# ===== Load generator model
modelG = model_setup_G_rulesD.Generator()
modelG.load_state_dict(torch.load('simple_dynamics_generator.pt'))


# ===== Generate several sample outputs
nGSample = 10
zLatent = torch.randn([nGSample, model_setup_G_rulesD.inputDimG])
yGen = modelG(zLatent)
data_size = 100
# x_get = model_setup_G_rulesD.data_generator(data_size)
# epoch_loss = 0
# y_gG1 = torch.Tensor(x_get)

# ===== Calculate Lyapunov equation satisfaction
nTimeStamps = 100
nFeatures = 200
for m1 in range(0, nGSample):

    y_ = yGen[m1]
    x1 = y_[0:nTimeStamps]
    x2 = y_[nTimeStamps:nFeatures]
    v = torch.zeros([100, 1])
    penalty_lyp = torch.zeros([100, 1])

    #---- v is the precalculated lyapunov candidae function x'Px
    for m in range(1, nTimeStamps, 1):
        v[m] = 3.1 * (x1[m]**2) + 0.2 * (x1[m]*x2[m]) + 0.6 * (x2[m]**2)
        penalty_lyp[m] = ((v[m]-v[m-1])+abs(v[m]-v[m-1]))

    if torch.mean(penalty_lyp) <= 0.001:
        print('Lyapunov equation satisfied', torch.mean(penalty_lyp))

    else:
        print('Lyapunov equation not satisfied', torch.mean(penalty_lyp))




# ===== Calculate distances among samples (diversity)
for m1 in range(0, nGSample):
    for m2 in range(m1, nGSample):
        dist = ell2_distance(yGen[m1].detach().numpy(), yGen[m2].detach().numpy())
print('Distance between samples is:', dist)

# ===== Plot a few sample G outputs
# npRow = 5
# npCol = 2
# timePlot = torch.linspace(0, 10, model_setup.nTimeStamps).to("cpu")
# px = 1/plt.rcParams['figure.dpi']
# fig, ax = plt.subplots(npRow, npCol)
# fig.set_figwidth(1800*px)
# fig.set_figheight(1500*px)
# modelG.to("cpu")
# for m1 in range(0, npRow):
#     for m2 in range(0, npCol):
#         x1_plt = yGen[m1 * npCol + m2][0:model_setup.nTimeStamps].detach().numpy()
#         x2_plt = yGen[m1 * npCol + m2][model_setup.nTimeStamps:model_setup.nFeatures].detach().numpy()
#         ax[m1, m2].plot(timePlot, x1_plt)
#         ax[m1, m2].plot(timePlot, x2_plt)
#
# plt.show()


# # ===== Load examples from test data
# nTestData = 10
# testDataset = np.genfromtxt('Data/lti1d_uncertainA_all.csv', delimiter=',')
# print(np.size(testDataset, 0))
# print(np.size(testDataset, 1))

# print(testDataset[0])
# Calculate sample distances to test data (similarity)
# for m1 in range(0, nGSample):
   # for m2 in range(0, nTestData):
        # ell2_distance(yGen[m1].detach().numpy(), yTest)