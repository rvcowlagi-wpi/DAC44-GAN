# ========== Dependencies
import torch
import torch.nn as nn
import numpy as np

# ===== Problem size
nFeatures = 200
nTimeStamps = 100
dt_ = 0.1

# ========== Multi-layer generator model
# ===== Generator model size
inputDimG = 100
dimG1 = 256
dimG2 = 512
dimG3 = 1024
dimG4 = 2048
dimG5 = 4096
dimG6 = 2048
dimG7 = 1024
dimG8 = 512
dimG9 = 256
dimG10 = 256


# ===== Torch NN class setup for generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(inputDimG, dimG1),  # input layer
            nn.LeakyReLU(0.2),
            nn.Linear(dimG1, dimG2),  # hidden layer 1
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG2, dimG3),  # hidden layer 2
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG3, dimG4),  # hidden layer 3
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG4, dimG5),  # hidden layer 4
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG5, dimG5),  # hidden layer 5
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG5, dimG6),  # hidden layer 5
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG6, dimG6),  # hidden layer 6
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG6, dimG7),  # hidden layer 7
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG7, dimG8),  # hidden layer 8
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG8, dimG9),  # hidden layer 9
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG9, dimG10),  # hidden layer 10
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dimG10, nFeatures)  # output layer
        )

    def forward(self, x):
        return self.layers(x)


# # define our data generation function
# def data_generator(data_size):
#     # f(x) = x_dot = A * x  ====> A = [[0,1],[-k1, -k2]] where k1>0, k2>0
#     inputs = []
#
#     for ix in range(data_size):
#
#         x1 = np.random.randint(1000) / 1000  # torch.randn
#         x2 = np.random.randint(1000) / 1000
#
#         # === calculate the x_dot value using the function
#         # a = [[0, 1], [-1.5, -0.5]]
#         # x_dot = a * x
#         x1_dot = x2
#         x2_dot = (-1.5*x1) + (-0.5*x2)
#         xdot = [x1_dot, x2_dot]
#
#         inputs.append([xdot])
#         # labels.append([y])
#
#     return inputs


# ========== Rules discriminator: straight lines
tmp1 = torch.arange(0, nFeatures, 1)
tmp2 = torch.ones(nFeatures)
C = torch.vstack((tmp1, tmp2))
C = torch.transpose(C, 0, 1)


def rules_d_straight_line(y_, device):
    C.to(device)
    x_ = torch.linalg.lstsq(C, y_).solution
    return torch.norm(y_ - torch.matmul(C, x_))


# =========== Rules discriminator: LTI state model
def rules_d_state1(y_, device):
    x1_ = y_[0:nTimeStamps]
    x2_ = y_[nTimeStamps:nFeatures]

    penalty_ = torch.zeros([nFeatures, 1]).to(device)  # this is just for proper sizing

    for m in range(1, nTimeStamps, 1):
        penalty_[m] = 30 * (10 * (x1_[m] - x1_[m - 1]) - x2_[m - 1])
        penalty_[m + nTimeStamps] = 30 * (10 * (x2_[m] - x2_[m - 1]) + 5 * x1_[m - 1] + x2_[m - 1])

    return penalty_





