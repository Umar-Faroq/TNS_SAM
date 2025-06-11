import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import cv2
from segment_anything import sam_model_registry
from skimage import transform, io
import torch.nn.functional as F
from glob import glob
import pandas as pd
all_dice, all_iou, all_precision, all_recall, all_specificity, all_accuracy = [], [], [], [], [], []
# ============ CONFIG ============
image_path = "/home/dilab/ext_drive/Thyroid_Nodule_segmentation/Thyroid_Dataset/TN3K/test-image/"
mask_path = "/home/dilab/ext_drive/Thyroid_Nodule_segmentation/Thyroid_Dataset/TN3K/test-mask/"
save_path = "./tn3k_infer_results"
model_ckpt = "/home/dilab/ext_drive/Thyroid_Nodule_segmentation/MICCAI2025/Comparision/MedSAM/work_dir/medsam_vit_b.pth"
device = "cuda:0"

os.makedirs(save_path, exist_ok=True)
image_list = sorted(glob(os.path.join(image_path, "*.jpg")))
mask_list = sorted(glob(os.path.join(mask_path, "*.jpg")))

# ============ LOAD MODEL ============
medsam_model = sam_model_registry["vit_b"](checkpoint=model_ckpt)
medsam_model = medsam_model.to(device)
medsam_model.eval()
def compute_metrics(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    TP = np.logical_and(pred == 1, gt == 1).sum()
    TN = np.logical_and(pred == 0, gt == 0).sum()
    FP = np.logical_and(pred == 1, gt == 0).sum()
    FN = np.logical_and(pred == 0, gt == 1).sum()

    epsilon = 1e-8
    dice = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
    iou = (TP + epsilon) / (TP + FP + FN + epsilon)
    precision = (TP + epsilon) / (TP + FP + epsilon)
    recall = (TP + epsilon) / (TP + FN + epsilon)
    specificity = (TN + epsilon) / (TN + FP + epsilon)
    accuracy = (TP + TN + epsilon) / (TP + TN + FP + FN + epsilon)

    return dice, iou, precision, recall, specificity, accuracy
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

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    return (low_res_pred > 0.5).astype(np.uint8)


# ============ INFERENCE LOOP ============
for img_fp, msk_fp in zip(image_list, mask_list):
    print(f"Running: {os.path.basename(img_fp)}")

    img = io.imread(img_fp)
    # msk = io.imread(msk_fp, as_gray=True)
    # # print(f"{os.path.basename(msk_fp)} mask shape: {msk.shape}, unique values: {np.unique(msk)}")
    # # msk = (msk > 127).astype(np.uint8)
    # msk = io.imread(msk_fp)
    # if msk.ndim == 3:
    #     msk = msk[:, :, 0]  # take red channel (assumed binary)
    # msk = (msk > 0).astype(np.uint8)
    
    msk = cv2.imread(msk_fp, cv2.IMREAD_GRAYSCALE)



    # Robust binarization
    msk = (msk > 10).astype(np.uint8)

    print(f"{os.path.basename(msk_fp)} -> after fix | shape: {msk.shape}, unique: {np.unique(msk)}")


    H, W = img.shape[:2]
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)

    # resize image to 1024x1024
    img_1024 = transform.resize(img, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_norm = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    img_tensor = torch.tensor(img_norm).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # compute bounding box from mask
    y_indices, x_indices = np.where(msk > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        print("⚠️ Skipping due to empty mask.")
        print(f"❌ Skipping {os.path.basename(img_fp)} | Mask unique: {np.unique(msk)} | Shape: {msk.shape}")

        continue
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    box_np = np.array([[x_min, y_min, x_max, y_max]])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    with torch.no_grad():
        img_embed = medsam_model.image_encoder(img_tensor)

    pred_mask = medsam_inference(medsam_model, img_embed, box_1024, H, W)
    # plt.imshow(pred_mask, cmap='gray')
    # plt.title(f"Prediction: {os.path.basename(img_fp)}")
    # plt.axis('off')
    # plt.show()
    dice, iou, prec, rec, spec, acc = compute_metrics(pred_mask, msk)
    print(f"✅ Processed {os.path.basename(img_fp)} | Dice: {dice:.4f}")

    all_dice.append(dice)
    all_iou.append(iou)
    all_precision.append(prec)
    all_recall.append(rec)
    all_specificity.append(spec)
    all_accuracy.append(acc)

    # save_name = f"medsam_{os.path.basename(img_fp)}"
    # save_fp = os.path.join(save_path, save_name)
    # io.imsave(save_fp, pred_mask.astype(np.uint8), check_contrast=False)
    # print(f"✅ Saved: {save_fp}")
    save_name = f"medsam_{os.path.basename(img_fp)}"
    save_fp = os.path.join(save_path, save_name)
    io.imsave(save_fp, (pred_mask * 255).astype(np.uint8), check_contrast=False)
    print(f"✅ Saved: {save_fp}")



if len(all_dice) == 0:
    print("❌ No masks were successfully processed. Please check mask thresholding or input paths.")
    exit()

print("\n📊 Evaluation Metrics on TN3K Test Set:")
print(f"  Dice (DSC):     {np.mean(all_dice):.4f}")
print(f"  IoU:            {np.mean(all_iou):.4f}")
print(f"  Precision:      {np.mean(all_precision):.4f}")
print(f"  Recall:         {np.mean(all_recall):.4f}")
print(f"  Specificity:    {np.mean(all_specificity):.4f}")
print(f"  Accuracy:       {np.mean(all_accuracy):.4f}")

# Save all metrics to CSV
results_df = pd.DataFrame({
    "Image": [os.path.basename(fp) for fp in image_list[:len(all_dice)]],
    "DSC": all_dice,
    "IoU": all_iou,
    "Precision": all_precision,
    "Recall": all_recall,
    "Specificity": all_specificity,
    "Accuracy": all_accuracy
})

csv_path = os.path.join(save_path, "medsam_eval_metrics.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n📄 Metrics saved to: {csv_path}")
