# ========== Dependencies
import torch
import torch.nn as nn

# ========== Multi-layer generator model
# ===== Generator model size
nFeatures = 200
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
            nn.ELU(),
            nn.Linear(dimG1, dimG2),  # hidden layer 1
            nn.ELU(),
            nn.Linear(dimG2, dimG3),  # hidden layer 2
            nn.ELU(),
            nn.Linear(dimG3, dimG4),  # hidden layer 3
            nn.ELU(),
            nn.Linear(dimG4, dimG5),  # hidden layer 4
            nn.ELU(),
            nn.Linear(dimG5, dimG5),  # hidden layer 5
            nn.ELU(),
            nn.Linear(dimG5, dimG6),  # hidden layer 5
            nn.ELU(),
            nn.Linear(dimG6, dimG6),  # hidden layer 6
            nn.ELU(),
            nn.Linear(dimG6, dimG7),  # hidden layer 7
            nn.ELU(),
            nn.Linear(dimG7, dimG8),  # hidden layer 8
            nn.ELU(),
            nn.Linear(dimG8, dimG9),  # hidden layer 9
            nn.ELU(),
            nn.Linear(dimG9, dimG10),  # hidden layer 10
            nn.ELU(),
            nn.Linear(dimG10, nFeatures)  # output layer
        )

    def forward(self, x):
        return self.layers(x)


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
n_time = round(nFeatures / 2)
dt = 0.1


def rules_d_state1(y_, device):
    x1_ = y_[0:n_time]
    x2_ = y_[n_time:nFeatures]

    penalty_ = torch.zeros([nFeatures, 1]).to(device)  # this is just for proper sizing

    for m in range(1, n_time, 1):
        penalty_[m] = 30 * (10 * (x1_[m] - x1_[m - 1]) - x2_[m - 1])
        penalty_[m + n_time] = 30 * (10 * (x2_[m] - x2_[m - 1]) + 5 * x1_[m - 1])

    return penalty_
