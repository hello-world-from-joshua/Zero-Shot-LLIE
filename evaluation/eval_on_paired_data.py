# ========================================================================
# Evalution code for paired (LOLv1 and LOLv2) datasets.
# To use, please replace input_dirs and ground_truth_dirs with your paths.
# Command: python eval_on_paired_data.py
# ========================================================================

import os
import cv2
import math
import pyiqa
import numpy as np
from datetime import datetime
from skimage import img_as_ubyte

import torch


def load_image(image_path, resize=None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize and (image.shape[0] != resize[1] or image.shape[1] != resize[0]):
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image /= 255.0
    return image

# This function is sourced from the following:
# Cai, Yuanhao et al. Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement, ICCV 2023.
def calculate_psnr(img1, img2):
    mse_ = np.mean((img1 - img2) ** 2)
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)

# This function is sourced from the following:
# Cai, Yuanhao et al. Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement, ICCV 2023.
def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError(f"Input images must have the same dimensions")
    
    ssims = []
    for i in range(3):
        ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
    return np.array(ssims).mean()

# This function is sourced from the following:
# Cai, Yuanhao et al. Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement, ICCV 2023.
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

lpips_metric = pyiqa.create_metric("lpips", device=device)
ms_ssim_metric = pyiqa.create_metric("ms_ssim", device=device)
vif_metric = pyiqa.create_metric("vif", device=device)
fsim_metric = pyiqa.create_metric("fsim", device=device)
dists_metric = pyiqa.create_metric("dists", device=device)

def convert_to_tensor(image):
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)

# Replace below strings with actual directory paths
input_dirs = [
    "path to KinD output for LOLv1 or LOLv2 Real",
    "path to KinD++ output for LOLv1 or LOLv2 Real",
    "path to SNR output for LOLv1 or LOLv2 Real",
    "path to GSAD output for LOLv1 or LOLv2 Real",
    "path to Retinexformer output for LOLv1 or LOLv2 Real",
    "path to RUAS output for LOLv1 or LOLv2 Real",
    "path to SCI output for LOLv1 or LOLv2 Real",
    "path to Zero-DCE output for LOLv1 or LOLv2 Real",
    "path to GDP output for LOLv1 or LOLv2 Real",
    "path to Our output for LOLv1 or LOLv2 Real"
]

# Replace below strings with actual directory paths
ground_truth_dirs = [
    "path to ground-truth directory",
    "path to ground-truth directory",
    "path to ground-truth directory",
    "path to ground-truth directory",
    "path to ground-truth directory",
    "path to ground-truth directory",
    "path to ground-truth directory",
    "path to ground-truth directory",
    "path to ground-truth directory",
    "path to ground-truth directory",
]

for input_dir, ground_truth_dir in zip(input_dirs, ground_truth_dirs):
    psnr_list = []
    ssim_list = []
    lpips_list = []
    ms_ssim_list = []
    vif_list = []
    fsim_list = []
    dists_list = []

    for filename in os.listdir(ground_truth_dir):
        img_gt_path = os.path.join(ground_truth_dir, filename)
        img_study_path = os.path.join(input_dir, filename)

        gt_img = load_image(img_gt_path)
        study_img = load_image(img_study_path, resize=(gt_img.shape[1], gt_img.shape[0]))

        gt_tensor = convert_to_tensor(gt_img)
        study_tensor = convert_to_tensor(study_img)

        psnr_list.append(calculate_psnr(gt_img, study_img))
        ssim_list.append(calculate_ssim(img_as_ubyte(gt_img), img_as_ubyte(study_img)))
        lpips_list.append(lpips_metric(gt_tensor, study_tensor).item())
        ms_ssim_list.append(ms_ssim_metric(gt_tensor, study_tensor).item())
        vif_list.append(vif_metric(gt_tensor, study_tensor).item())
        fsim_list.append(fsim_metric(gt_tensor, study_tensor).item())
        dists_list.append(dists_metric(gt_tensor, study_tensor).item())

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    avg_ms_ssim = np.mean(ms_ssim_list)
    avg_vif = np.mean(vif_list)
    avg_fsim = np.mean(fsim_list)
    avg_dists = np.mean(dists_list)

    print(f"\nResults for {input_dir}:")
    print(f"Average PSNR: {avg_psnr:.3f}")
    print(f"Average SSIM: {avg_ssim:.3f}")
    print(f"Average LPIPS: {avg_lpips:.3f}")
    print(f"Average MS-SSIM: {avg_ms_ssim:.3f}")
    print(f"Average VIF: {avg_vif:.3f}")
    print(f"Average FSIM: {avg_fsim:.3f}")
    print(f"Average DISTS: {avg_dists:.3f}")