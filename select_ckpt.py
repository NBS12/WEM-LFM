import torch

# 你的 ckpt 文件路径
ckpt_path = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/logs/2026-05-03T07-19-36_train/checkpoints/last.ckpt"

# 加载 checkpoint
checkpoint = torch.load(ckpt_path, map_location="cpu")

print("checkpoint 中的 key：")
print(checkpoint.keys())

# 常见字段
if "epoch" in checkpoint:
    print(f"\n训练轮数 epoch: {checkpoint['epoch']}")

if "global_step" in checkpoint:
    print(f"全局 step: {checkpoint['global_step']}")

# 如果是 pytorch-lightning
if "loops" in checkpoint:
    print("\n这是 PyTorch Lightning checkpoint")

# 打印完整结构（可选）
# print(checkpoint)