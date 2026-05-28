import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from skimage import transform
import SimpleITK as sitk
from segment_anything import sam_model_registry

# ================== 路径配置（按你给的信息已填好） ==================
REAL_DIR = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/images/test"
GT_MASK_DIR = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/masks/test"

FAKE_DIR = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model/test_results_7.5"
BOX_DIR = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/box"  # {maskid}_bbox.nii.gz

OUT_DIR = "./viz_real_fake_medsam"
MEDSAM_CKPT = "medsam_vit_b.pth"  # 改成你的绝对路径也可以
DEVICE = "cuda:0"

IMG_SIZE = 256
# ===================================================================


def read_nifti_arr(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return np.squeeze(arr)


def bbox_from_binary_mask(mask_hw: np.ndarray):
    """mask_hw: (H,W) binary {0,1} -> return (x0,y0,x1,y1) in ORIGINAL coords"""
    ys, xs = np.where(mask_hw > 0)
    if len(xs) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1, y1


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024_xyxy, out_h=256, out_w=256, thr=0.5):
    """box_1024_xyxy: (1,4) [x0,y0,x1,y1] on 1024 scale"""
    box_torch = torch.as_tensor(box_1024_xyxy, dtype=torch.float32, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B,1,4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    prob = torch.sigmoid(low_res_logits)
    prob = F.interpolate(prob, size=(out_h, out_w), mode="bilinear", align_corners=False)
    prob = prob.squeeze().detach().cpu().numpy()
    seg = (prob > thr).astype(np.uint8)
    return seg


@torch.no_grad()
def segment_image_with_given_box(img_path: str, box_xyxy_orig, medsam_model, device=DEVICE):
    """
    img_path: png
    box_xyxy_orig: (x0,y0,x1,y1) in ORIGINAL coords (H,W from bbox nii)
    Return:
      seg_256 (256,256) {0,1}
      img_256_rgb (256,256,3) uint8
    """
    # image 256 rgb
    img_256_rgb = np.array(Image.open(img_path).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).convert("RGB"), dtype=np.uint8)

    # to 1024 and normalize [0,1]
    img_1024 = transform.resize(img_256_rgb, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    img_1024_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # scale bbox orig->1024 using original H,W from bbox file later
    # 这里不做缩放，外面会传 box_1024
    img_embed = medsam_model.image_encoder(img_1024_t)
    return img_embed, img_256_rgb


def draw_bbox(img_rgb, box_256, color=(0, 255, 0), thickness=2):
    x0, y0, x1, y1 = [int(v) for v in box_256]
    out = img_rgb.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    return out


def overlay_gt_pred(img_rgb, gt_lesion01, pred_lesion01):
    """
    GT lesion -> green channel
    Pred lesion -> red channel
    overlap -> yellow
    """
    out = img_rgb.copy()
    pred = (pred_lesion01 > 0).astype(np.uint8) * 255
    gt = (gt_lesion01 > 0).astype(np.uint8) * 255

    out[..., 0] = np.maximum(out[..., 0], pred)  # R
    out[..., 1] = np.maximum(out[..., 1], gt)    # G
    return out


def main(max_cases=30, thr=0.5):
    os.makedirs(OUT_DIR, exist_ok=True)

    # load medsam
    medsam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_CKPT).to(DEVICE).eval()

    names = sorted(os.listdir(GT_MASK_DIR))
    used = 0

    for name in names:
        if not name.lower().endswith(".png"):
            continue
        maskid = name.split(".")[0]

        # paths
        gt_path = os.path.join(GT_MASK_DIR, name)
        real_path = os.path.join(REAL_DIR, name)
        fake_path = os.path.join(FAKE_DIR, name)
        box_path = os.path.join(BOX_DIR, f"{maskid}_bbox.nii.gz")

        if not os.path.exists(real_path) or not os.path.exists(fake_path) or not os.path.exists(box_path):
            continue

        # GT lesion (class 2)
        gt_mask = np.array(Image.open(gt_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST).convert("L"), dtype=np.uint8)
        gt_lesion = (gt_mask == 2).astype(np.uint8)

        # 只看含病灶病例（和你一致）
        if gt_lesion.sum() == 0:
            continue

        # read bbox nii to get original H,W and box coords
        bbox_arr = read_nifti_arr(box_path)
        if bbox_arr.ndim != 2:
            continue
        H, W = bbox_arr.shape
        bbox_bin = (bbox_arr > 0).astype(np.uint8)

        box_orig = bbox_from_binary_mask(bbox_bin)
        if box_orig is None:
            continue
        x0, y0, x1, y1 = box_orig
        box_xyxy_orig = np.array([x0, y0, x1, y1], dtype=np.float32)

        # scale to 256/1024 (IMPORTANT: xyxy uses [W,H,W,H])
        box_256 = box_xyxy_orig / np.array([W, H, W, H], dtype=np.float32) * IMG_SIZE
        box_1024 = box_xyxy_orig / np.array([W, H, W, H], dtype=np.float32) * 1024.0
        box_1024 = box_1024[None, :]  # (1,4)

        # prepare out folder for this case
        case_dir = os.path.join(OUT_DIR, maskid)
        os.makedirs(case_dir, exist_ok=True)

        # -------- real --------
        real_embed, real_img256 = segment_image_with_given_box(real_path, box_xyxy_orig, medsam_model, DEVICE)
        real_pred = medsam_inference(medsam_model, real_embed, box_1024, out_h=IMG_SIZE, out_w=IMG_SIZE, thr=thr)

        # -------- fake --------
        fake_embed, fake_img256 = segment_image_with_given_box(fake_path, box_xyxy_orig, medsam_model, DEVICE)
        fake_pred = medsam_inference(medsam_model, fake_embed, box_1024, out_h=IMG_SIZE, out_w=IMG_SIZE, thr=thr)

        # 1) bbox-only
        real_bbox_img = draw_bbox(real_img256, box_256, color=(0, 255, 0), thickness=2)
        fake_bbox_img = draw_bbox(fake_img256, box_256, color=(0, 255, 0), thickness=2)

        # 2) overlay GT + pred + bbox
        real_ov = overlay_gt_pred(real_bbox_img, gt_lesion, real_pred)
        fake_ov = overlay_gt_pred(fake_bbox_img, gt_lesion, fake_pred)

        Image.fromarray(real_bbox_img).save(os.path.join(case_dir, "real_bbox.png"))
        Image.fromarray(fake_bbox_img).save(os.path.join(case_dir, "fake_bbox.png"))
        Image.fromarray(real_ov).save(os.path.join(case_dir, "real_overlay.png"))
        Image.fromarray(fake_ov).save(os.path.join(case_dir, "fake_overlay.png"))

        # also save raw masks (optional)
        Image.fromarray((gt_lesion * 255).astype(np.uint8)).save(os.path.join(case_dir, "gt_lesion.png"))
        Image.fromarray((real_pred * 255).astype(np.uint8)).save(os.path.join(case_dir, "real_pred.png"))
        Image.fromarray((fake_pred * 255).astype(np.uint8)).save(os.path.join(case_dir, "fake_pred.png"))

        used += 1
        if used >= max_cases:
            break

    print(f"Saved {used} cases to: {OUT_DIR}")
    print(f"Tip: compare real_overlay.png vs fake_overlay.png under the same case folder.")
    print(f"If you want tighter masks to improve IoU, rerun with thr=0.6/0.65.")


if __name__ == "__main__":
    # max_cases: 先看30例足够定位问题；thr: 0.5是默认，想压缩红区可改0.6
    main(max_cases=30, thr=0.5)
