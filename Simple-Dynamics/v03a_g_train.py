import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time
from datetime import datetime


device = "cuda"

start_clock = time.time()

n_features = 200
my_batch_size = 1

# ----- Create generator model class
# --- Worked for straight line
# input_dimG = 25
# dimG_1 = 32
# dimG_2 = 64
# dimG_3 = 256
# dimG_4 = 1024
# dimG_5 = 256
# dimG_6 = 64
# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.layers = nn.Sequential(
#             nn.Linear(input_dimG, dimG_1),  # input layer
#             nn.ReLU(),
#             nn.Linear(dimG_1, dimG_2),  # hidden layer 1
#             nn.ReLU(),
#             nn.Linear(dimG_2, dimG_3),  # hidden layer 2
#             nn.ReLU(),
#             nn.Linear(dimG_3, dimG_4),  # hidden layer 3
#             nn.ReLU(),
#             nn.Linear(dimG_4, dimG_5),  # hidden layer 4
#             nn.ReLU(),
#             nn.Linear(dimG_5, dimG_6),  # hidden layer 5
#             nn.ReLU(),
#             nn.Linear(dimG_6, n_features)  # output layer
#         )
#
#     def forward(self, x):
#         return self.layers(x)

# --- Somewhat worked for x1_dot and x2_dot
input_dimG = 100
dimG_1 = 256
dimG_2 = 512
dimG_3 = 1024
dimG_4 = 2048
dimG_5 = 4096
dimG_6 = 2048
dimG_7 = 1024
dimG_8 = 512
dimG_9 = 256
dimG_10= 256
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dimG, dimG_1),  # input layer
            nn.ELU(),
            nn.Linear(dimG_1, dimG_2),  # hidden layer 1
            nn.ELU(),
            nn.Linear(dimG_2, dimG_3),  # hidden layer 2
            nn.ELU(),
            nn.Linear(dimG_3, dimG_4),  # hidden layer 3
            nn.ELU(),
            nn.Linear(dimG_4, dimG_5),  # hidden layer 4
            nn.ELU(),
            nn.Linear(dimG_5, dimG_5),  # hidden layer 5
            nn.ELU(),
            nn.Linear(dimG_5, dimG_6),  # hidden layer 5
            nn.ELU(),
            nn.Linear(dimG_6, dimG_6),  # hidden layer 6
            nn.ELU(),
            nn.Linear(dimG_6, dimG_7),  # hidden layer 7
            nn.ELU(),
            nn.Linear(dimG_7, dimG_8),  # hidden layer 8
            nn.ELU(),
            nn.Linear(dimG_8, dimG_9),  # hidden layer 9
            nn.ELU(),
            nn.Linear(dimG_9, dimG_10),  # hidden layer 10
            nn.ELU(),
            nn.Linear(dimG_10, n_features)  # output layer
        )

    def forward(self, x):
        return self.layers(x)








# # Somewhat worked for x1_dot and x2_dot
# input_dimG = 100
# dimG_1 = 256
# dimG_2 = 512
# dimG_3 = 1024
# dimG_4 = 2048
# dimG_5 = 4096
# dimG_6 = 2048
# dimG_7 = 1024
# dimG_8 = 512
# dimG_9 = 256
# dimG_10= 256
# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.layers = nn.Sequential(
#             nn.Linear(input_dimG, dimG_1),  # input layer
#             nn.ReLU(),
#             nn.Linear(dimG_1, dimG_2),  # hidden layer 1
#             nn.ReLU(),
#             nn.Linear(dimG_2, dimG_3),  # hidden layer 2
#             nn.ReLU(),
#             nn.Linear(dimG_3, dimG_4),  # hidden layer 3
#             nn.ReLU(),
#             nn.Linear(dimG_4, dimG_5),  # hidden layer 4
#             nn.ReLU(),
#             nn.Linear(dimG_5, dimG_5),  # hidden layer 5
#             nn.ReLU(),
#             nn.Linear(dimG_5, dimG_5),  # hidden layer 6
#             nn.ReLU(),
#             nn.Linear(dimG_5, dimG_6),  # hidden layer 7
#             nn.ReLU(),
#             nn.Linear(dimG_6, dimG_6),  # hidden layer 8
#             nn.ReLU(),
#             nn.Linear(dimG_6, dimG_6),  # hidden layer 9
#             nn.ReLU(),
#             nn.Linear(dimG_6, dimG_7),  # hidden layer 10
#             nn.ReLU(),
#             nn.Linear(dimG_7, dimG_7),  # hidden layer 11
#             nn.ReLU(),
#             nn.Linear(dimG_7, dimG_8),  # hidden layer 12
#             nn.ReLU(),
#             nn.Linear(dimG_8, dimG_9),  # hidden layer 13
#             nn.ReLU(),
#             nn.Linear(dimG_9, dimG_10),  # hidden layer 14
#             nn.ReLU(),
#             nn.Linear(dimG_10, n_features)  # output layer
#         )
#
#     def forward(self, x):
#         return self.layers(x)




# # Continuity rules discrominator
# def RulesDiscriminator(y_gen):
#
#     solution_found_ = 0
#
#     for m in range(99):
#         slope1 = (y_gen[m-1]-y_gen[m])/0.1
#         slope2 = (y_gen[m]-y_gen[m+1])/0.1
#         slope_check = abs(slope1-slope2)
#
#         # out = self.sigmoid(slope_check)
#
#     if slope_check <= 1E-1:
#         solution_found_ = 1
#
#     print(slope_check)
#
#     return solution_found_, slope_check


# Straight line rules discriminator
tmp1 = torch.arange(0, n_features, 1).to(device)
tmp2 = torch.ones(n_features).to(device)
C = torch.vstack((tmp1, tmp2))
C = torch.transpose(C, 0, 1)
def StraightLineRulesD(y_):
    x_ = torch.linalg.lstsq(C, y_).solution
    return torch.norm( y_ - torch.matmul(C, x_))


# n_time = round(n_features/2)
# dt = 0.1
# def LyapnuovRulesD1(y_):
#     x1_ = y_[0:n_time]
#     x2_ = y_[n_time:n_features]
#
#     # print(x1_)
#     # print(x2_)
#
#
#     penalty_ = 0.1*(x1_[0] - 1)*(x1_[0] - 1)
#     for m in range(n_time):
#
#         dx1 = (x1_[m] - x1_[m-1])
#         pen1 = (10*dx1 - x2_[m-1])
#         # pen2 = (x2_[m] - 1)
#
#         dx2 = (x2_[m] - x2_[m - 1])
#         pen2 = dx2 #10*dx2 + x1_[m-1]
#         # pen2 = x2_[m]
#
#         # Vm   = (x2_[m]*x2_[m])
#         # Vmm1 = (x2_[m-1]*x2_[m-1])
#         # pen2  = (Vm - Vmm1)
#         penalty_ = penalty_ + 0.1*pen1*pen1 + 0.01*(pen2 - 10)*(pen2 - 10)
#
#     rule_ = penalty_ #1 / (1 + torch.exp(-(penalty_/30 - 6)))
#
#     return rule_



n_time = round(n_features/2)
dt = 0.1
def LyapnuovRulesD1(y_):
    x1_ = y_[0:n_time]
    x2_ = y_[n_time:n_features]

    pen1 = torch.zeros([n_features,1]).to(device)  # this is just for proper sizing

    # pen1[0] = 0.1*(x1_[0] - 1)
    # pen1[n_time] = 0.1*(x2_[0] - 3)
    for m in range(1, n_time, 1):
        pen1[m] = 30*(10*(x1_[m] - x1_[m - 1]) - x2_[m-1])
        pen1[m + n_time] = 30*(10*(x2_[m] - x2_[m - 1]) + 5*x1_[m-1])


    return pen1



# Instantiate generator
modelG = Generator()
modelG.to(device)


# Instantiate loss and optimizer
loss_dg = nn.MSELoss().to(device)

learning_rate_gen = 2E-4
optimizerG = torch.optim.Adam(modelG.parameters(), lr=learning_rate_gen)


n_epochs = 1000
n_iter = 0
G_losses = []
D_losses = []
D_losses_test = []
D_out_test = []
rules_out = []
for epoch in range(n_epochs):
        print('\n======== Epoch ', epoch, ' ========')

        # GENERATOR
        # Reset gradients
        modelG.zero_grad()
        optimizerG.zero_grad()

        # Sample from latent vector space
        z = torch.randn([my_batch_size, input_dimG]).to(device)

        # Run latent vector through G
        y_gG = modelG(z).to(device)

        error_of_d2 = LyapnuovRulesD1(y_gG.reshape([n_features, 1]))
        # print(error_of_d2)
        # error_of_d2 = StraightLineRulesD(y_gG.reshape([n_features, 1]))

        errorG2 = loss_dg(error_of_d2, Variable(torch.zeros(n_features, 1)).to(device))
        errorG2.backward()

        # Update G
        optimizerG.step()

        n_iter += 1
        # Save Losses for plotting later
        G_losses.append(errorG2.item())
        print(errorG2.item())

        # if (epoch > 10) and abs(G_losses[epoch] - G_losses[epoch-1]) < 1E-4:
        #     break

        # #y_d_fakeG = modelD(y_gG)  # Since we just updated D, perform another forward pass of all-fake batch through D
        #
        # # solution_found, slope_check = RulesDiscriminator(y_gG.reshape([100, 1]))

        # print(error_of_d2)


        # torch.tensor(solution_found, dtype=torch.int)
        # print(solution_found)
        # if solution_found == 1:
        #     d2 = 0
        # else:
        #     d2 = 1
        #     error_of_d2 = slope_check.detach()  # 0.1
        # Using D2
        # print(y_gG.size())


        # Calculate error and backpropagate """
        #print(y_d_fakeG.size())
        #error_of_d2.reshape([1, 1])
        #print(error_of_d2.size())
        # errorG = loss_dg(y_d_fakeG, ones_target(my_batch_size))

        # error_gd1d2 = (w1 * errorG) + (w2 * errorG2)



        # loss = criterion(inputs, y_gG)  # Difference between real and fake trajectory points



        # D_out_test.append(y_d_fakeG.item())
        # rules_out.append(solution_found)
        # D_losses.append(errorD.item())
        # D_losses.append(error_of_d2.item())

        # if n_iter == (len(train_loader.dataset) / n_features) * n_epochs:  # i % 5 == 0:
        #     print('Iteration: {}. Loss: {}'.format(n_iter, abs(loss.item())))
        #     Disc_percentage = torch.mean(y_d_fakeG)
        #     print('Discriminator output: this is {} % real'.format(100 - (Disc_percentage.item() * 100)))


# # for i, (test_traj) in enumerate(test_loader):
# #     # Test with real examples from dataset
# #     # print(traj)
# #     inputs_test = test_traj.float()  # .to(device)
# #     # inputs = np.array(inputs, dtype=np.float32)
# #
# #     y_d_test = modelD(inputs_test)
# #
# #     # Calculate error and backpropagate
# #     #error_test = loss_dg(y_d_test, ones_target(len(test_traj)))
# #     #D_losses_test.append(error_test.item())
# #     D_out_test.append(y_d_test.item())

stop_clock = time.time()
elapsed_hours = int((stop_clock - start_clock)//3600)
elapsed_minutes = int((stop_clock - start_clock)//60 - elapsed_hours*60)
elapsed_seconds = (stop_clock - start_clock) - elapsed_hours*3600 - elapsed_minutes*60

print('\nElapsed time ' + str(elapsed_hours) + ':' + str(elapsed_minutes) + ':' + "%.2f" % elapsed_seconds)

# Plot a few sample generator results
n_plot_row = 5
n_plot_col = 2
# fig2 = plt.figure()
fig, ax = plt.subplots(n_plot_row, n_plot_col)

# --- For 2 states
t = torch.linspace(0, 10, n_time).to("cpu")
px = 1/plt.rcParams['figure.dpi']
fig.set_figwidth(1800*px)
fig.set_figheight(1500*px)
modelG.to("cpu")
for m1 in range(0, n_plot_row):
    for m2 in range(0, n_plot_col):
        z = torch.randn([my_batch_size, input_dimG])
        y = modelG(z)

        x1_plt = y[0][0:n_time].detach().numpy()
        x2_plt = y[0][n_time:n_features].detach().numpy()
        ax[m1, m2].plot(t, x1_plt)
        ax[m1, m2].plot(t, x2_plt)


# --- For 1 state
# t = torch.linspace(0, 10, n_features).to("cpu")
# px = 1/plt.rcParams['figure.dpi']
# fig.set_figwidth(900*px)
# fig.set_figheight(1500*px)
# modelG.to("cpu")
# for m1 in range(0, n_plot_row):
#     for m2 in range(0, n_plot_col):
#         z = torch.randn([my_batch_size, input_dimG])
#         y = modelG(z)
#
#         x1_plt = y[0].detach().numpy()
#         ax[m1].plot(t, x1_plt)


# --- Generator loss
# fig3 = plt.figure()
# plt.plot(G_losses, label="G_loss")
# plt.title("Generator loss during training")
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.legend()


ts = (np.datetime64(datetime.utcnow()) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
figname_ = "Results/202305/rules_only_" + str(round(ts)) + ".png"
fig.savefig(figname_, bbox_inches='tight')

plt.show()