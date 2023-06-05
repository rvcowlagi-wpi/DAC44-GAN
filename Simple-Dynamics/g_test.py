# Script for testing a generator model.
# Takes several sample G outputs and compares against a library
# using L2 distance as a similarity test.
# Also checks for diversity of samples, i.e., whether the distance between samples is zeros.


# ========== Dependencies
import torch
import numpy as np
import model_setup


# =========== L2 distance calculator
def ell2_distance(z1_, z2_):
    # z1 and z2 are two vectors of equal size
    distance_ = np.sum(np.square(z1_ - z2_))
    return distance_


# ===== Load generator model
modelG = model_setup.Generator()
modelG.load_state_dict(torch.load('simple_dynamics_generator.pt'))

# ===== Generate several sample outputs
nGSample = 100
zLatent = torch.randn([nGSample, model_setup.inputDimG])
yGen = modelG(zLatent)

# ===== Calculate distances among samples (diversity)
for m1 in range(0, nGSample):
    for m2 in range(m1, nGSample):
        ell2_distance(yGen[m1].detach().numpy(), yGen[m2].detach().numpy())


# ===== Load examples from test data
nTestData = 10

# ===== Calculate sample distances to test data (similarity)
# for m1 in range(0, nGSample):
#     for m2 in range(0, nTestData):
#         ell2_distance(yGen[m1].detach().numpy(), yTest)