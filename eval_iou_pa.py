import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import cv2
from segment_anything import sam_model_registry
from skimage import transform


# ==============================
# 工具函数
# ==============================
def Read_nifti(img_path):
    img_sitk = sitk.ReadImage(img_path)
    return sitk.GetArrayFromImage(img_sitk)


def breast_mask_from_u8_eval(img_u8: np.ndarray, T: int = 15) -> np.ndarray:
    img_u8 = img_u8.astype(np.uint8)

    mask = np.zeros_like(img_u8, dtype=np.uint8)
    mask[img_u8 > T] = 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

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

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False
    )
    return (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def seg_inference(data_path, mask_path, medsam_model, device):
    img_3c = np.array(
        Image.open(data_path).resize((256, 256)).convert("RGB")
    )

    img_1024 = transform.resize(img_3c, (1024, 1024), preserve_range=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / max(img_1024.max() - img_1024.min(), 1e-8)

    img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    bbox_arr = Read_nifti(mask_path)
    H, W = bbox_arr.shape

    xs = np.where(bbox_arr == 1)[0]
    ys = np.where(bbox_arr == 1)[1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    box = np.array([[ymin, xmin, ymax, xmax]], dtype=np.float32)
    box_1024 = box / np.array([W, H, W, H]) * 1024.0

    image_embedding = medsam_model.image_encoder(img_tensor)

    return medsam_inference(medsam_model, image_embedding, box_1024, 256, 256)


# ==============================
# 分批生成分割图
# ==============================
def generate_seg_batch(mask_path, fake_path, box_path, device, start, end):
    print(f"处理范围: {start} ~ {end}")

    mask_dict = {}
    for name in os.listdir(box_path):
        maskid = name.split("_bbox")[0]
        mask_dict.setdefault(maskid, []).append(name)

    medsam_model = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth").to(device)
    medsam_model.eval()

    savedir = fake_path + "_segs_new"
    os.makedirs(savedir, exist_ok=True)

    names = sorted(os.listdir(mask_path))[start:end]

    for idx, name in enumerate(names):
        print(f"{start+idx}: {name}")

        save_path = os.path.join(savedir, name)
        if os.path.exists(save_path):
            continue

        gt_mask = np.array(
            Image.open(os.path.join(mask_path, name))
            .resize((256, 256), Image.NEAREST)
            .convert("L")
        )

        if (gt_mask == 2).sum() == 0:
            continue

        img_path = os.path.join(fake_path, name)

        img = np.array(
            Image.open(img_path).resize((256, 256)).convert("L")
        ).astype(np.uint8)

        breast = breast_mask_from_u8_eval(img)

        seg_mask = np.zeros_like(img)
        seg_mask[breast > 0] = 1

        maskid = name.split(".")[0]
        if maskid in mask_dict:
            for box_name in mask_dict[maskid]:
                boxpath = os.path.join(box_path, box_name)
                medsam_seg = seg_inference(img_path, boxpath, medsam_model, device)
                seg_mask[medsam_seg == 1] = 2

        Image.fromarray(seg_mask.astype(np.uint8)).save(save_path)

    print("这一批完成")

class SegmentationMetric:
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((numClass, numClass))

    def pixelAccuracy(self):
        return np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()

    def classPixelAccuracy(self):
        return np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)

    def meanPixelAccuracy(self):
        return np.nanmean(self.classPixelAccuracy())

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = (
            self.confusionMatrix.sum(axis=1)
            + self.confusionMatrix.sum(axis=0)
            - intersection
        )
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return IoU, mIoU

    def addBatch(self, pred, gt):
        mask = (gt >= 0) & (gt < self.numClass)
        label = self.numClass * gt[mask].astype(int) + pred[mask].astype(int)
        count = np.bincount(label, minlength=self.numClass ** 2)
        self.confusionMatrix += count.reshape(self.numClass, self.numClass)
# ==============================
# 快速计算 IoU / PA
# ==============================
def compute_IoU_PA_from_saved_seg(mask_path, pred_path):
    metric = SegmentationMetric(3)

    valid_count = 0
    skip_count = 0

    for name in os.listdir(mask_path):
        gt_file = os.path.join(mask_path, name)
        pred_file = os.path.join(pred_path, name)

        if not os.path.exists(pred_file):
            skip_count += 1
            continue

        gt = np.array(Image.open(gt_file).resize((256, 256)))
        pred = np.array(Image.open(pred_file).resize((256, 256)))

        if (gt == 2).sum() == 0:
            skip_count += 1
            continue

        gt[gt > 2] = 0
        pred[pred > 2] = 0

        metric.addBatch(pred, gt)
        valid_count += 1

    print("valid_count:", valid_count)
    print("skip_count:", skip_count)

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
# 主函数
# ==============================
if __name__ == "__main__":
    mask_path = "/dev/raid/zjs_dc3/24sxx/data/masks/test"
    fake_path = "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/results/test_results_lesion_aware_frequency_mask_pramv4.0"
    box_path = "/dev/raid/zjs_dc3/24sxx/data/box"

    device = "cuda:0"

    # ====== 每次改这里 ======
    #generate_seg_batch(mask_path, fake_path, box_path, device, start=700, end=1872)

    # ====== 全部生成完再跑这个 ======
    compute_IoU_PA_from_saved_seg(mask_path, fake_path + "_segs_new")