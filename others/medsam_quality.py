import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from skimage import transform
import SimpleITK as sitk
from segment_anything import sam_model_registry


# -----------------------------
# NIfTI / BBox utilities
# -----------------------------
def read_nifti_arr(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return np.squeeze(arr)

def bbox_from_binary_mask(mask_hw: np.ndarray):
    """mask_hw: (H,W) with {0,1}. Return (x0,y0,x1,y1) in ORIGINAL coords."""
    ys, xs = np.where(mask_hw > 0)
    if len(xs) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1, y1


# -----------------------------
# MedSAM inference
# -----------------------------
@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024_xyxy, out_h=256, out_w=256, thr=0.5):
    """
    box_1024_xyxy: np.array shape (1,4) in [x0,y0,x1,y1] on 1024 scale
    return: seg (out_h,out_w) {0,1}
    """
    box_torch = torch.as_tensor(box_1024_xyxy, dtype=torch.float32, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B,1,4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    prob = torch.sigmoid(low_res_logits)  # (B,1,256,256) typically
    prob = F.interpolate(prob, size=(out_h, out_w), mode="bilinear", align_corners=False)
    prob = prob.squeeze().detach().cpu().numpy()
    seg = (prob > thr).astype(np.uint8)
    return seg

@torch.no_grad()
def medsam_segment_with_bbox(real_img_path, bbox_nii_path, medsam_model, device="cuda:0"):
    """
    real_img_path: real mammogram image (png/jpg)
    bbox_nii_path: bbox mask nifti used to prompt MedSAM
    return: pred_lesion_256 (256,256) {0,1}, img_256_rgb (256,256,3)
    """
    # image -> 256 rgb
    img_256_rgb = np.array(
        Image.open(real_img_path).resize((256, 256), Image.BILINEAR).convert("RGB"),
        dtype=np.uint8
    )

    # image -> 1024 and normalize to [0,1]
    img_1024 = transform.resize(
        img_256_rgb, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    img_1024_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # bbox from nifti (original H,W)
    bbox_arr = read_nifti_arr(bbox_nii_path)
    if bbox_arr.ndim != 2:
        raise ValueError(f"Expected 2D bbox array, got {bbox_arr.shape} from {bbox_nii_path}")
    H, W = bbox_arr.shape
    bbox_bin = (bbox_arr > 0).astype(np.uint8)

    box = bbox_from_binary_mask(bbox_bin)
    if box is None:
        return np.zeros((256, 256), dtype=np.uint8), img_256_rgb

    x0, y0, x1, y1 = box
    box_xyxy_orig = np.array([[x0, y0, x1, y1]], dtype=np.float32)
    box_1024 = box_xyxy_orig / np.array([W, H, W, H], dtype=np.float32) * 1024.0

    # encode and segment
    img_embed = medsam_model.image_encoder(img_1024_t)
    pred = medsam_inference(medsam_model, img_embed, box_1024, out_h=256, out_w=256, thr=0.5)
    return pred, img_256_rgb


# -----------------------------
# Metrics (lesion only + optional 3-class)
# -----------------------------
def iou_binary(pred01: np.ndarray, gt01: np.ndarray):
    pred = (pred01 > 0).astype(np.uint8)
    gt = (gt01 > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)

def dice_binary(pred01: np.ndarray, gt01: np.ndarray):
    pred = (pred01 > 0).astype(np.uint8)
    gt = (gt01 > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter) / float(denom)

def save_overlay(img_rgb_256: np.ndarray, pred01: np.ndarray, gt01: np.ndarray, save_path: str):
    """
    overlay:
      GT lesion in green, Pred lesion in red (overlap looks yellowish).
    """
    out = img_rgb_256.copy()
    pred = (pred01 > 0).astype(np.uint8) * 255
    gt = (gt01 > 0).astype(np.uint8) * 255

    # BGR in cv2, but we're using RGB array; do channel-wise max manually.
    # pred -> red channel, gt -> green channel
    out[..., 0] = np.maximum(out[..., 0], pred)  # R
    out[..., 1] = np.maximum(out[..., 1], gt)    # G
    Image.fromarray(out).save(save_path)


# -----------------------------
# Main: Upper bound test on REAL images
# -----------------------------
def medsam_upperbound_real(
    real_img_dir: str,
    gt_mask_dir: str,
    bbox_dir: str,
    medsam_ckpt: str = "medsam_vit_b.pth",
    device: str = "cuda:0",
    save_vis: bool = True,
    vis_dir: str = "./medsam_real_vis",
):
    """
    Assumptions (matching你的项目)：
      - real_img_dir 下的图像文件名与 gt_mask_dir 的 mask 文件名一致（如 xxx.png）
      - gt_mask 是 0/1/2，其中病灶类=2
      - bbox_dir 里是 {maskid}_bbox*.nii 或 nii.gz
    """
    os.makedirs(vis_dir, exist_ok=True)

    # build bbox dict
    bbox_dict = {}
    for fn in os.listdir(bbox_dir):
        if "_bbox" not in fn:
            continue
        maskid = fn.split("_bbox")[0]
        bbox_dict.setdefault(maskid, []).append(fn)

    # load medsam
    medsam_model = sam_model_registry["vit_b"](checkpoint=medsam_ckpt).to(device).eval()

    names = sorted(os.listdir(gt_mask_dir))
    lesion_ious = []
    lesion_dices = []
    used = 0
    skipped_no_mass = 0
    skipped_missing_bbox = 0
    skipped_missing_img = 0

    for name in names:
        gt_path = os.path.join(gt_mask_dir, name)
        gt_mask = np.array(
            Image.open(gt_path).resize((256, 256), Image.NEAREST).convert("L"),
            dtype=np.uint8
        )
        gt_lesion = (gt_mask == 2).astype(np.uint8)

        if gt_lesion.sum() == 0:
            skipped_no_mass += 1
            continue

        img_path = os.path.join(real_img_dir, name)
        if not os.path.exists(img_path):
            skipped_missing_img += 1
            continue

        maskid = name.split(".")[0]
        if maskid not in bbox_dict:
            skipped_missing_bbox += 1
            continue

        # run medsam for each bbox and OR them (更稳：多框时合并)
        pred_lesion = np.zeros((256, 256), dtype=np.uint8)
        img_rgb_256 = None
        for bbox_fn in bbox_dict[maskid]:
            bbox_path = os.path.join(bbox_dir, bbox_fn)
            pred_one, img_rgb_256 = medsam_segment_with_bbox(img_path, bbox_path, medsam_model, device=device)
            pred_lesion = np.maximum(pred_lesion, pred_one.astype(np.uint8))

        iou = iou_binary(pred_lesion, gt_lesion)
        dice = dice_binary(pred_lesion, gt_lesion)
        lesion_ious.append(iou)
        lesion_dices.append(dice)
        used += 1

        if save_vis and img_rgb_256 is not None:
            save_overlay(
                img_rgb_256, pred_lesion, gt_lesion,
                os.path.join(vis_dir, name.replace(".png", "_overlay.png"))
            )

    # summary
    if used == 0:
        print("No cases evaluated. Check paths / filenames / bbox dict.")
        return

    print("---- MedSAM Upper Bound on REAL images ----")
    print(f"used_cases(with mass) = {used}")
    print(f"skipped_no_mass       = {skipped_no_mass}")
    print(f"skipped_missing_img   = {skipped_missing_img}")
    print(f"skipped_missing_bbox  = {skipped_missing_bbox}")
    print(f"lesion IoU  mean = {np.mean(lesion_ious):.6f} | std = {np.std(lesion_ious):.6f}")
    print(f"lesion IoU median= {np.median(lesion_ious):.6f}")
    print(f"lesion Dice mean = {np.mean(lesion_dices):.6f} | std = {np.std(lesion_dices):.6f}")
    print(f"lesion Dice median= {np.median(lesion_dices):.6f}")
    if save_vis:
        print(f"Saved overlays to: {vis_dir}")


if __name__ == "__main__":
    # 改成你自己的路径（按你当前项目的目录）
    real_img_dir = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/images/test"
    gt_mask_dir  = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/masks/test"
    bbox_dir     = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/box"

    medsam_upperbound_real(
        real_img_dir=real_img_dir,
        gt_mask_dir=gt_mask_dir,
        bbox_dir=bbox_dir,
        medsam_ckpt="medsam_vit_b.pth",
        device="cuda:0",
        save_vis=True,
        vis_dir="./medsam_real_vis"
    )
