import model_setup


# ===== Initialize
device = model_setup.device




# ===== Initialize training
nEpochs = 3000
nIter = 0
lossRecordEn = []

# ===== Train autoencoder
for epoch in range(nEpochs):
    print('\n======== Epoch ', epoch, ' ========')
    for i, (traj) in enumerate(model_setup.trainLoader):

        theDecoder.zero_grad()
        theEncoder.zero_grad()

        # input real examples from dataset
        zReal = traj.float().to(device)

        # ----- Pass real data through encoder
        yEncoder = theEncoder(zReal)

        # ----- Pass encoder output through generator (generator acts as decoder)
        yGenerator = theDecoder(yEncoder).to(device)

        # Difference between generated and real data
        aeError = zReal - yGenerator

        # ----- Find gradient and update D
        aeLoss = loss_encoder(aeError, model_setup.zeros_target(nFeatures))
        aeLoss.backward()
        optimizerE.step()
        optimizerG.step()

        # ----- Book-keeping
        nIter += 1

        # Save Losses for plotting later
        lossRecordEn.append(aeLoss.item())