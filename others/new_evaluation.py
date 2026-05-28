import os
import math
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from skimage import transform
import SimpleITK as sitk
from segment_anything import sam_model_registry

# -----------------------------
# FID (same as yours)
# -----------------------------
def compute_fid(data_folder1, data_folder2, device='cuda:0'):
    from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
    feat_model = build_feature_extractor("clean", device, use_dataparallel=False)

    ref_features = get_folder_features(
        data_folder1, model=feat_model, num_workers=0, num=None,
        shuffle=False, seed=0, batch_size=32, device=torch.device(device),
        mode="clean", custom_fn_resize=None, description="", verbose=True,
        custom_image_tranform=None
    )
    mu_r, sigma_r = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)

    gen_features = get_folder_features(
        data_folder2, model=feat_model, num_workers=0, num=None,
        shuffle=False, seed=0, batch_size=32, device=torch.device(device),
        mode="clean", custom_fn_resize=None, description="", verbose=True,
        custom_image_tranform=None
    )
    mu_g, sigma_g = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

    score = frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    print(f"fid={score:.4f}")


# -----------------------------
# Mask helpers
# -----------------------------
def img2mask(img_gray, T=15):
    """
    Your original breast mask extractor:
    threshold + morphology + keep max connected component
    Returns {0,1} mask.
    """
    img = img_gray.copy()
    img[img < T] = 0
    img[img > 0] = 1
    mask = img.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        # nothing found -> return all zeros
        return np.zeros_like(mask, dtype=np.uint8)

    area = [cv2.contourArea(c) for c in contours]
    max_idx = int(np.argmax(area))

    # fill non-max components with 0
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(mask, [contours[k]], 0)

    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask.astype(np.uint8)


def read_nifti_arr(img_path):
    """
    Read nii/nii.gz and return squeezed numpy array.
    """
    img_sitk = sitk.ReadImage(img_path)
    arr = sitk.GetArrayFromImage(img_sitk)
    return np.squeeze(arr)


def bbox_from_binary_mask(bbox_arr_2d):
    """
    bbox_arr_2d shape: (H,W), binary {0,1}
    Return (x0,y0,x1,y1) in ORIGINAL coords.
    """
    ys, xs = np.where(bbox_arr_2d == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1, y1


# -----------------------------
# MedSAM inference
# -----------------------------
@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024_xyxy, out_h, out_w):
    """
    box_1024_xyxy: np array shape (1,4) in [x0,y0,x1,y1] on 1024 scale
    """
    box_torch = torch.as_tensor(box_1024_xyxy, dtype=torch.float32, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B,1,4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B,256,64,64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    prob = torch.sigmoid(low_res_logits)  # (B,1,256,256) typically
    prob = F.interpolate(prob, size=(out_h, out_w), mode="bilinear", align_corners=False)
    prob = prob.squeeze().detach().cpu().numpy()
    seg = (prob > 0.5).astype(np.uint8)
    return seg


@torch.no_grad()
def seg_inference(img_path, bbox_nii_path, medsam_model, device):
    """
    Runs MedSAM on a 256x256 image with bbox from bbox_nii_path.
    Returns:
      medsam_seg: (256,256) {0,1}
      img_256_rgb: (256,256,3) uint8
      box_256_xyxy: (1,4) float on 256 scale
    """
    # 1) image 256 RGB
    img_256_rgb = np.array(
        Image.open(img_path).resize((256, 256), Image.BILINEAR).convert('RGB'),
        dtype=np.uint8
    )

    # 2) image to 1024 and normalize [0,1]
    img_1024 = transform.resize(
        img_256_rgb, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)

    img_1024_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # 3) load bbox mask and compute box in ORIGINAL coords
    bbox_arr = read_nifti_arr(bbox_nii_path)
    if bbox_arr.ndim != 2:
        raise ValueError(f"Expected 2D bbox array, got {bbox_arr.shape} from {bbox_nii_path}")
    H, W = bbox_arr.shape
    bbox_bin = (bbox_arr == 1).astype(np.uint8)

    box = bbox_from_binary_mask(bbox_bin)
    if box is None:
        # no bbox pixels -> return empty seg
        empty = np.zeros((256, 256), dtype=np.uint8)
        return empty, img_256_rgb, np.array([[0, 0, 0, 0]], dtype=np.float32)

    x0, y0, x1, y1 = box
    box_np_xyxy = np.array([[x0, y0, x1, y1]], dtype=np.float32)

    # 4) scale to 256 and 1024
    box_256 = box_np_xyxy / np.array([W, H, W, H], dtype=np.float32) * 256.0
    box_1024 = box_np_xyxy / np.array([W, H, W, H], dtype=np.float32) * 1024.0

    # 5) image embedding
    img_embed = medsam_model.image_encoder(img_1024_t)  # (1,256,64,64)

    # 6) inference
    medsam_seg = medsam_inference(medsam_model, img_embed, box_1024, 256, 256)
    return medsam_seg, img_256_rgb, box_256

def keep_cc_nearest_bbox_center(bin_mask: np.ndarray, box_256: np.ndarray) -> np.ndarray:
    """
    bin_mask: 0/1 二值mask (256x256)
    box_256: shape (1,4) -> [ymin, xmin, ymax, xmax] in 256 coord  (与你 seg_inference 返回一致)

    返回：只保留“离 bbox 中心最近”的那个连通域后的 0/1 mask
    """
    if bin_mask is None or bin_mask.sum() == 0:
        return bin_mask

    # bbox center
    ymin, xmin, ymax, xmax = box_256[0]
    cy = 0.5 * (ymin + ymax)
    cx = 0.5 * (xmin + xmax)

    mask_u8 = (bin_mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        # 只有背景
        return mask_u8

    # centroids: (num_labels, 2) -> (x, y)
    best_label = None
    best_dist2 = None

    for lab in range(1, num_labels):  # 跳过背景 0
        x, y = centroids[lab]
        dist2 = (x - cx) ** 2 + (y - cy) ** 2
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_label = lab

    out = (labels == best_label).astype(np.uint8)
    return out


# -----------------------------
# Metrics
# -----------------------------
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2, dtype=np.float64)

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / np.clip(self.confusionMatrix.sum(), 1e-12, None)
        return acc

    def classPixelAccuracy(self):
        # NOTE: this is recall = TP/(TP+FN)
        classAcc = np.diag(self.confusionMatrix) / np.clip(self.confusionMatrix.sum(axis=1), 1e-12, None)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = (np.sum(self.confusionMatrix, axis=1) +
                 np.sum(self.confusionMatrix, axis=0) -
                 np.diag(self.confusionMatrix))
        IoU = intersection / np.clip(union, 1e-12, None)
        mIoU = np.nanmean(IoU)
        return IoU, mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # imgPredict, imgLabel are 2D arrays with values in [0, numClass-1]
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype(int) + imgPredict[mask].astype(int)
        count = np.bincount(label, minlength=self.numClass**2)
        return count.reshape(self.numClass, self.numClass)

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix[:] = 0


# -----------------------------
# Main evaluation
# -----------------------------
def compute_IoU_PA(mask_path, fake_path, box_path, device='cuda:0', save_vis=False):
    """
    mask_path: GT mask png folder (values expected 0/1/2)
    fake_path: generated images folder (same file names)
    box_path : bbox nii.gz folder with names like {maskid}_bbox*.nii.gz
    """
    # build dict: maskid -> list of bbox nii names
    mask_dict = {}
    for name in os.listdir(box_path):
        if "_bbox" not in name:
            continue
        maskid = name.split('_bbox')[0]
        mask_dict.setdefault(maskid, []).append(name)

    # load MedSAM
    MedSAM_CKPT_PATH = "medsam_vit_b.pth"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device).eval()

    # output segs
    savedir = fake_path + '_segs_new1'
    os.makedirs(savedir, exist_ok=True)
    visdir = fake_path + "_vis"
    if save_vis:
        os.makedirs(visdir, exist_ok=True)

    metric = SegmentationMetric(3)

    names = sorted(os.listdir(mask_path))
    used = 0
    skipped_no_mass = 0
    skipped_missing_img = 0
    skipped_missing_box = 0

    for name in names:
        gt_mask = np.array(
            Image.open(os.path.join(mask_path, name)).resize((256, 256), Image.NEAREST).convert('L'),
            dtype=np.uint8
        )

        # only evaluate images that contain mass in GT (same as your logic)
        if (gt_mask == 2).sum() == 0:
            skipped_no_mass += 1
            continue

        img_path = os.path.join(fake_path, name)
        if not os.path.exists(img_path):
            skipped_missing_img += 1
            continue

        maskid = name.split('.')[0]
        if maskid not in mask_dict:
            skipped_missing_box += 1
            continue

        # 1) breast base mask (0/1)
        img_gray = np.array(
            Image.open(img_path).resize((256, 256), Image.BILINEAR).convert('L'),
            dtype=np.uint8
        )
        seg_mask = img2mask(img_gray)  # {0,1}

        # 2) MedSAM lesion seg(s) using each bbox
        img_rgb_256 = None
        for box_name in mask_dict[maskid]:
            boxpath = os.path.join(box_path, box_name)

            medsam_seg, _, box_256 = seg_inference(img_path, boxpath, medsam_model, device)

            if medsam_seg is None:
                continue

    # 1) 可选：先做连通域筛选（最关键）
            medsam_seg = keep_cc_nearest_bbox_center(medsam_seg, box_256)

    # 2) 写入 lesion 类别（2）
            seg_mask[medsam_seg == 1] = 2

        # 3) IMPORTANT FIX:
        # addBatch(imgPredict, imgLabel) -> (pred, gt)
        metric.addBatch(seg_mask, gt_mask)
        used += 1

        # save predicted seg mask
        Image.fromarray(seg_mask.astype(np.uint8)).save(os.path.join(savedir, name))

        # optional visualization
        if save_vis and img_rgb_256 is not None:
            # overlay lesion
            ov = img_rgb_256.copy()
            lesion = (seg_mask == 2).astype(np.uint8) * 255
            ov[..., 0] = np.maximum(ov[..., 0], lesion)  # add red
            Image.fromarray(ov).save(os.path.join(visdir, name.replace('.png', '_overlay.png')))

    # report
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    iou, miou = metric.meanIntersectionOverUnion()

    print("---- Consistency (Pred vs GT) ----")
    print(f"used_cases(with mass) = {used}")
    print(f"skipped_no_mass       = {skipped_no_mass}")
    print(f"skipped_missing_img   = {skipped_missing_img}")
    print(f"skipped_missing_box   = {skipped_missing_box}")
    print(f"pa   = {pa:.6f}")
    print(f"cpa  = {cpa}")   # per-class recall
    print(f"mpa  = {mpa:.6f}")
    print(f"IoU  = {iou}")
    print(f"mIoU = {miou:.6f}")


if __name__ == '__main__':
    real_path = r'/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/images/test'
    mask_path = r'/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/masks/test'
    box_path  = r'/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-master/data/box'

    device = "cuda:0"
    fake = r'/home/zjs/sxx24/Gated-Conditional-Diffusion-Model/results/test_results_7.5'

    compute_fid(real_path, fake, device=device)
    compute_IoU_PA(mask_path, fake, box_path, device=device, save_vis=False)
