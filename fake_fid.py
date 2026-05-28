import numpy as np
import torch
from cleanfid.fid import get_folder_features, build_feature_extractor

fake_path = r"/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/results/test_results_lesion_aware_frequency_mask_pramv4.0"
device = "cuda:0"

feat_model = build_feature_extractor("clean", device, use_dataparallel=False)

features = get_folder_features(
    fake_path,
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

np.save("fake_features.npy", features)
print("生成图特征已保存到 fake_features.npy")
print(features.shape)