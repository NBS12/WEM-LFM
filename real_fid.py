import numpy as np
import torch
from cleanfid.fid import get_folder_features, build_feature_extractor

real_path = r"/dev/raid/zjs_dc3/24sxx/data/images/test"
device = "cuda:0"

feat_model = build_feature_extractor("clean", device, use_dataparallel=False)

features = get_folder_features(
    real_path,
    model=feat_model,
    num_workers=0,
    num=None,
    shuffle=False,
    seed=0,
    batch_size=32,
    device=torch.device(device),
    mode="clean",
    verbose=True,
)

np.save("real_features.npy", features)
print("真实图特征已保存到 real_features.npy")
print(features.shape)