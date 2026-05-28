
import os
import math
import random

import cv2
import lpips
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry
from skimage import io, transform
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.utils import make_grid



def compute_fid(data_folder1, data_folder2, device="cuda:0"):
    from cleanfid.fid import (
        get_folder_features,
        build_feature_extractor,
        frechet_distance,
    )

    feat_model = build_feature_extractor("clean", device, use_dataparallel=False)

    ref_features = get_folder_features(
        data_folder1,
        model=feat_model,
        num_workers=0,
        num=None,
        shuffle=False,
        seed=0,
        batch_size=32,
        device=torch.device(device),
        mode="clean",
        custom_fn_resize=None,
        description="",
        verbose=True,
        custom_image_tranform=None,
    )
    a2b_ref_mu, a2b_ref_sigma = (
        np.mean(ref_features, axis=0),
        np.cov(ref_features, rowvar=False),
    )

    gen_features = get_folder_features(
        data_folder2,
        model=feat_model,
        num_workers=0,
        num=None,
        shuffle=False,
        seed=0,
        batch_size=32,
        device=torch.device(device),
        mode="clean",
        custom_fn_resize=None,
        description="",
        verbose=True,
        custom_image_tranform=None,
    )
    ed_mu, ed_sigma = (
        np.mean(gen_features, axis=0),
        np.cov(gen_features, rowvar=False),
    )
    score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
    print(f"fid={score_fid_a2b:.4f}")


# ==============================
# 乳腺整体掩码（评估用，与预处理逻辑一致）
# ==============================
def breast_mask_from_u8_eval(img_u8: np.ndarray, T: int = 15) -> np.ndarray:
    """
    评估阶段的乳腺整体掩码：
    - 阈值分割 (T=15，与原始 img2mask 保持一致的大致范围)
    - 不做腐蚀，避免把乳头和乳房主体掰断
    - 只保留最大连通域
    - 再用闭运算填洞、平滑边界
    输入:  img_u8: uint8, 0~255
    输出:  mask: 0/1
    """
    img_u8 = img_u8.astype(np.uint8)

    # 1. 固定阈值分割：乳腺区域大致为 1，背景为 0
    mask = np.zeros_like(img_u8, dtype=np.uint8)
    mask[img_u8 > T] = 1

    # 2. 只保留最大连通域（此时乳头通常还与主体连在一起）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if num_labels > 1:
        # stats[0] 是背景，从 1 开始才是真正的连通域
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    # 3. 闭运算填洞和平滑轮廓（不会明显缩小或掰断区域）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask.astype(np.uint8)




# ==============================
# MedSAM 推理
# ==============================
@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,  # 和论文 evaluation.py 保持一致
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, H, W)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (H, W)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)  # 0/1

    return medsam_seg


def Read_nifti(img_path):
    img_sitk = sitk.ReadImage(img_path)
    return sitk.GetArrayFromImage(img_sitk)


def seg_inference(data_path, mask_path, medsam_model, device):
    """
    data_path: fake 图路径
    mask_path: 对应 bbox 的 nii.gz 路径
    返回: medsam_seg (256x256,0/1), img_3c(256x256x3), box_256(1,4)
    """
    img_3c = np.array(
        Image.open(data_path)
        .resize((256, 256), Image.BILINEAR)
        .convert("RGB")
    )

    # resize 到 1024x1024 并归一化
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    # 从 nii.gz bbox mask 中取出 box
    bbox_arr = Read_nifti(mask_path)  # 2D
    H, W = bbox_arr.shape
    xs = np.where(bbox_arr == 1)[0]
    ys = np.where(bbox_arr == 1)[1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    # 注意顺序: [ymin, xmin, ymax, xmax]
    box_np = np.array([[ymin, xmin, ymax, xmax]], dtype=np.float32)
    box_256 = box_np / np.array([W, H, W, H], dtype=np.float32) * 256.0
    box_1024 = box_np / np.array([W, H, W, H], dtype=np.float32) * 1024.0

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)

    medsam_seg = medsam_inference(
        medsam_model, image_embedding, box_1024, 256, 256
    )  # 0/1

    return medsam_seg, img_3c, box_256


# ==============================
# 分割指标
# ==============================
class SegmentationMetric(object):
    """
    confusionMatrix:
        P\L     P    N
        P      TP    FP
        N      FN    TN
    """

    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # overall pixel accuracy
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # per-class pixel accuracy
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = (
            np.sum(self.confusionMatrix, axis=1)
            + np.sum(self.confusionMatrix, axis=0)
            - np.diag(self.confusionMatrix)
        )
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return IoU, mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # imgPredict & imgLabel: same shape, 值域 [0, numClass-1]
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIoU = sum_i (freq_i * IoU_i)
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
            np.sum(self.confusionMatrix, axis=1)
            + np.sum(self.confusionMatrix, axis=0)
            - np.diag(self.confusionMatrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


# ==============================
# IoU / PA 计算
# ==============================
def compute_IoU_PA(mask_path, fake_path, box_path, device="cuda:0"):
    """
    mask_path: 真实 0/1/2 掩码目录 (GT)
    fake_path: 生成图目录
    box_path : nii.gz bbox 目录
    """
    # 收集每个 maskid 对应的所有 bbox nii
    mask_dict = {}
    for name in os.listdir(box_path):
        maskid = name.split("_bbox")[0]
        mask_dict.setdefault(maskid, []).append(name)

    # 加载 MedSAM
    MedSAM_CKPT_PATH = "medsam_vit_b.pth"
    medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    savedir = fake_path + "_segs_new"
    os.makedirs(savedir, exist_ok=True)

    metric = SegmentationMetric(3)

    for name in os.listdir(mask_path):
        gt_mask = np.array(
            Image.open(os.path.join(mask_path, name))
            .resize((256, 256), Image.NEAREST)
            .convert("L")
        )
        # 只评估含病灶的样本（和原代码保持一致）
        if (gt_mask == 2).sum() == 0:
            continue

        save = os.path.join(savedir, name)
        img_path = os.path.join(fake_path, name)

        # 读生成图 (256x256,灰度)
        img = np.array(
            Image.open(img_path)
            .resize((256, 256), Image.BILINEAR)
            .convert("L")
        ).astype(np.uint8)

        # 1) 用与预处理一致的方法生成乳腺 mask (0/1)
        breast = breast_mask_from_u8_eval(img)

        # 2) 初始化三类 seg_mask: 0=bg,1=breast,2=lesion
        seg_mask = np.zeros_like(img, dtype=np.uint8)
        seg_mask[breast > 0] = 1

        # 3) 对该图的所有 bbox 运行 MedSAM，将病灶填为 2
        maskid = name.split(".")[0]
        if maskid in mask_dict:
            for box_name in mask_dict[maskid]:
                boxpath = os.path.join(box_path, box_name)
                medsam_seg, _, _ = seg_inference(
                    img_path, boxpath, medsam_model, device
                )
                seg_mask[medsam_seg == 1] = 2

        # 4) 更新混淆矩阵，并保存可视化 mask
        #print(name, "unique seg_mask:", np.unique(seg_mask), "counts:", np.bincount(seg_mask.flatten(), minlength=3))
        metric.addBatch(seg_mask, gt_mask)
        Image.fromarray(seg_mask.astype(np.uint8)).save(save)

    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU, mIoU = metric.meanIntersectionOverUnion()
    print("pa is : %f" % pa)
    print("cpa is :")
    print(cpa)
    print("mpa is : %f" % mpa)
    print("Iou is :")
    print(IoU)
    print("mIoU is : %f" % mIoU)


# ==============================
# main
# ==============================
if __name__ == "__main__":
    real_path = r"/dev/raid/zjs_dc3/24sxx/data/images/test"
    mask_path = r"/dev/raid/zjs_dc3/24sxx/data/masks/test"
    box_path = r"/dev/raid/zjs_dc3/24sxx/data/box"

    device = "cuda:0"
    fake = r"/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/results/test_results_lesion_aware_frequency_mask_pramv4.0"

    compute_fid(real_path, fake, device=device)
    compute_IoU_PA(mask_path, fake, box_path, device=device)
