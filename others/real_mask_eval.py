import os
import numpy as np
from PIL import Image
import torch
import SimpleITK as sitk
from segment_anything import sam_model_registry
from skimage import transform
import torch.nn.functional as F

# -------------------------
# Utils
# -------------------------
def Read_nifti(img_path: str) -> np.ndarray:
    img_sitk = sitk.ReadImage(img_path)
    return sitk.GetArrayFromImage(img_sitk)

def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (inter / union) if union > 0 else 1.0

def find_same_name_image(folder: str, stem: str):
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None

# -------------------------
# MedSAM inference (same style as your evaluation.py)
# -------------------------
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
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def seg_inference_evalstyle(img_path: str, box_nii_path: str, medsam_model, device: str):
    # 1) read image -> 256 RGB
    img_3c = np.array(Image.open(img_path).resize((256, 256), Image.BILINEAR).convert("RGB"))

    # 2) build 1024 input (same as your evaluation.py)
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)

    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # 3) get bbox from nii mask
    bbox_arr = Read_nifti(box_nii_path)
    H, W = bbox_arr.shape

    xs = np.where(bbox_arr == 1)[0]
    ys = np.where(bbox_arr == 1)[1]
    if len(xs) == 0 or len(ys) == 0:
        return None

    xmin = xs.min()
    xmax = xs.max()
    ymin = ys.min()
    ymax = ys.max()

    # NOTE: keep consistent with your evaluation.py: [ymin, xmin, ymax, xmax]
    box_np = np.array([[ymin, xmin, ymax, xmax]], dtype=np.float32)
    box_1024 = box_np / np.array([W, H, W, H], dtype=np.float32) * 1024.0

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)

    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, 256, 256)
    return medsam_seg

# -------------------------
# Main
# -------------------------
def main():
    # ====== paths (edit if needed) ======
    real_img_dir = r"/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/images/test"
    gt_mask_dir  = r"/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/masks/test"
    box_dir      = r"/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/box"

    # saved fake segs from evaluation.py
    fake_segs_dir = r"/home/zjs/sxx24/Gated-Conditional-Diffusion-Model/results/test_results_7.5_segs"

    # output dir for real_pred_evalstyle
    out_real_pred_dir = r"/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/real_pred_evalstyle"
    os.makedirs(out_real_pred_dir, exist_ok=True)

    device = "cuda:0"
    MedSAM_CKPT_PATH = "medsam_vit_b.pth"

    # Build MedSAM (same as your evaluation.py)
    medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    # maskid -> [bbox_files]
    box_map = {}
    for fn in os.listdir(box_dir):
        if fn.endswith(".nii.gz") and "_bbox" in fn:
            maskid = fn.split("_bbox")[0]
            box_map.setdefault(maskid, []).append(os.path.join(box_dir, fn))

    # stats
    scores_rf = []  # IoU(real_pred_evalstyle, fake_lesion)
    scores_gr = []  # IoU(gt_lesion, real_pred_evalstyle)
    scores_gf = []  # IoU(gt_lesion, fake_lesion)

    missing_real_img = 0
    missing_fake_seg = 0
    missing_bbox = 0

    # iterate using gt masks as sample list (only those with lesion==2)
    for fn in sorted(os.listdir(gt_mask_dir)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        maskid = os.path.splitext(fn)[0]

        gt_path = os.path.join(gt_mask_dir, fn)
        gt = np.array(Image.open(gt_path).resize((256, 256), Image.NEAREST).convert("L"))
        gt_lesion = (gt == 2).astype(np.uint8)
        if gt_lesion.sum() == 0:
            continue

        real_img_path = find_same_name_image(real_img_dir, maskid)
        if real_img_path is None:
            missing_real_img += 1
            continue

        bbox_list = box_map.get(maskid, [])
        if len(bbox_list) == 0:
            missing_bbox += 1
            continue

        # 1) real_pred_evalstyle (union over multiple bboxes)
        real_union = np.zeros((256, 256), dtype=np.uint8)
        for bbox_path in bbox_list:
            pred = seg_inference_evalstyle(real_img_path, bbox_path, medsam_model, device)
            if pred is None:
                continue
            real_union[pred == 1] = 1

        # save real pred (0/255)
        real_pred_save = os.path.join(out_real_pred_dir, f"{maskid}_real_pred.png")
        Image.fromarray((real_union * 255).astype(np.uint8)).save(real_pred_save)

        # 2) read fake seg (3 classes) and extract lesion==2
        fake_seg_path = os.path.join(fake_segs_dir, f"{maskid}.png")
        if not os.path.exists(fake_seg_path):
            missing_fake_seg += 1
            continue
        fake_seg = np.array(Image.open(fake_seg_path).resize((256, 256), Image.NEAREST).convert("L"))
        fake_lesion = (fake_seg == 2).astype(np.uint8)

        # 3) compute three IoUs
        iou_rf = iou(real_union, fake_lesion)
        iou_gr = iou(gt_lesion, real_union)
        iou_gf = iou(gt_lesion, fake_lesion)

        scores_rf.append(iou_rf)
        scores_gr.append(iou_gr)
        scores_gf.append(iou_gf)

        print(
            f"{maskid}: "
            f"IoU(real_pred, fake_pred)={iou_rf:.4f} | "
            f"IoU(gt, real_pred)={iou_gr:.4f} | "
            f"IoU(gt, fake_pred)={iou_gf:.4f}"
        )

    # Summary
    print("\n==== Summary ====")
    print(f"Matched samples: {len(scores_rf)}")
    print(f"Missing real images: {missing_real_img}")
    print(f"Missing bbox: {missing_bbox}")
    print(f"Missing fake segs: {missing_fake_seg}")

    if len(scores_rf) > 0:
        print("\n---- Mean ----")
        print(f"mean IoU(real_pred, fake_pred): {float(np.mean(scores_rf)):.4f}")
        print(f"mean IoU(gt, real_pred):        {float(np.mean(scores_gr)):.4f}")
        print(f"mean IoU(gt, fake_pred):        {float(np.mean(scores_gf)):.4f}")

        print("\n---- Median ----")
        print(f"median IoU(real_pred, fake_pred): {float(np.median(scores_rf)):.4f}")
        print(f"median IoU(gt, real_pred):        {float(np.median(scores_gr)):.4f}")
        print(f"median IoU(gt, fake_pred):        {float(np.median(scores_gf)):.4f}")

        print("\n---- Min/Max ----")
        print(f"real_pred vs fake_pred: {float(np.min(scores_rf)):.4f} / {float(np.max(scores_rf)):.4f}")
        print(f"gt vs real_pred:        {float(np.min(scores_gr)):.4f} / {float(np.max(scores_gr)):.4f}")
        print(f"gt vs fake_pred:        {float(np.min(scores_gf)):.4f} / {float(np.max(scores_gf)):.4f}")
    else:
        print("No matched samples to compute IoU. Check filename rules and paths.")

if __name__ == "__main__":
    main()
