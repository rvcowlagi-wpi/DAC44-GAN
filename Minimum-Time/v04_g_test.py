# Script for testing a generator model.
# Takes several sample G outputs and compares against a library
# 1) L2 distance as a similarity test.
# 2) Checks for diversity of samples, i.e., whether the distance between samples is zeros.
# 3) Checks for Lyapunov stability,i.e., quadratic Lyapunov function decreases monotonically along the trajectory


# ========== Dependencies
import torch
import numpy as np
import model_setup
import numpy
import pandas

# ===== Initialize
device = model_setup.device
myBatchSize = model_setup.myBatchSize


# ===== Load generator model
theGenerator = model_setup.DecoderGenerator()
theGenerator.load_state_dict(torch.load('simple_dynamics_generator.pt'))


# ===== Generate several sample outputs
nGSample = 10
nLatent = torch.randn([myBatchSize, model_setup.dimLatent]).to(device)
zFake = theGenerator(nLatent).to(device)

# ========== Test rules with the ral data
# ========== Load the real training data
data2d = numpy.array(pandas.read_csv('Data/data2D.txt'))
trainLoader = torch.utils.data.DataLoader(data2d, batch_size=myBatchSize)
for i, (traj) in enumerate(model_setup.trainLoader):
    zReal = traj.float().to(device)


# =========== L2 distance calculator
def ell2_distance(z1_, z2_):
    # z1 and z2 are two vectors of equal size
    distance_ = np.sum(np.square(z1_ - z2_))
    return distance_


# ===== Calculate Lyapunov equation satisfaction
nTimeStamps = 100
nFeatures = 200
for m1 in range(0, nGSample):

    # y_ = zReal[m1]
    y_ = zFake[m1]
    y_ = y_.reshape([nFeatures, 1])
    x1 = y_[0:nTimeStamps]
    x2 = y_[nTimeStamps:nFeatures]
    v = torch.zeros([100, 1])
    penalty_lyp = torch.zeros([100, 1])

    # ---- v is the precalculated lyapunov candidae function x'Px
    for m in range(1, nTimeStamps, 1):
        v[m] = 3.1 * (x1[m]**2) + 0.2 * (x1[m]*x2[m]) + 0.6 * (x2[m]**2)
        # v[m] = 2.5 * (x1[m] ** 2) + 0.0 * (x1[m] * x2[m]) + 0.5 * (x2[m] ** 2)
        penalty_lyp[m] = v[m]-v[m-1]  # +abs(v[m]-v[m-1]))

    if torch.mean(penalty_lyp) <= 1e-3:
        print('Lyapunov equation satisfied', torch.mean(penalty_lyp))

    else:
        print('Lyapunov equation not satisfied', torch.mean(penalty_lyp))


# ===== Calculate distances among samples (diversity)
for m1 in range(0, nGSample):
    for m2 in range(m1, nGSample):
        dist = ell2_distance(zFake[m1].detach().numpy(), zFake[m2].detach().numpy())
print('Distance between samples is:', dist)







