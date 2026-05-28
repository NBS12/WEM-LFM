import os
import numpy as np
from PIL import Image

# =========================
# 输入路径
# =========================
mask_dir = "/dev/raid/zjs_dc3/24sxx/data/masks/test/"

# =========================
# 输出路径（你指定的）
# =========================
save_dir = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/results/mask_test_vis"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 颜色映射
# =========================
color_map = {
    0: (0, 0, 0),      # 背景
    1: (0, 255, 0),    # 乳腺
    2: (255, 0, 0),    # 病灶
}

# =========================
# 遍历处理
# =========================
saved_count = 0
skipped_count = 0

for file_name in os.listdir(mask_dir):
    if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")):
        continue

    file_path = os.path.join(mask_dir, file_name)

    try:
        # 读取mask
        if file_name.lower().endswith(".npy"):
            mask = np.load(file_path)
        else:
            mask = np.array(Image.open(file_path))

        # 如果是RGB，转单通道
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = mask.astype(np.uint8)

        # =========================
        # ⭐ 关键：只处理包含病灶的mask
        # =========================
        if not np.any(mask == 2):
            skipped_count += 1
            print(f"跳过（无病灶）: {file_name}")
            continue

        # =========================
        # 生成彩色mask
        # =========================
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for k, v in color_map.items():
            color_mask[mask == k] = v

        # 保存
        save_name = os.path.splitext(file_name)[0] + "_lesion.png"
        save_path = os.path.join(save_dir, save_name)
        Image.fromarray(color_mask).save(save_path)

        saved_count += 1
        print(f"已保存（含病灶）: {save_path}")

    except Exception as e:
        print(f"处理失败: {file_name}, error: {e}")

print("\n======================")
print(f"保存数量（含病灶）: {saved_count}")
print(f"跳过数量（无病灶）: {skipped_count}")
print("完成！")