import numpy as np
from cleanfid.fid import frechet_distance

real_features = np.load("real_features.npy")
fake_features = np.load("fake_features.npy")

real_mu = np.mean(real_features, axis=0)
real_sigma = np.cov(real_features, rowvar=False)

fake_mu = np.mean(fake_features, axis=0)
fake_sigma = np.cov(fake_features, rowvar=False)

fid = frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

print(f"FID: {fid:.4f}")