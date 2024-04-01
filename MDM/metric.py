import argparse
import glob
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--results")


def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def torch_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

if __name__ == "__main__":
    args = parser.parse_args()

    results = glob.glob(os.path.join(args.results, "*.jpg"))

    psnr_ls = []
    ssim_ls = []

    with tqdm(
        initial=0,
        total=len(results),
    ) as pbar:
        for res in results:
            concat_img = cv2.imread(res)
            # print(concat_img.shape)
            h, w, _ = concat_img.shape
            single_w = w // 3

            rs_img = concat_img[:, :single_w]
            gt_gs_img = concat_img[:, single_w : single_w * 2]
            our_gs_img = concat_img[:, single_w * 2 :]

            _psnr = psnr(gt_gs_img, our_gs_img)
            _ssim = ssim(gt_gs_img, our_gs_img, channel_axis=2)

            psnr_ls.append(_psnr)
            ssim_ls.append(_ssim)

            # print(f"PSNR {_psnr}/{np.mean(psnr_ls)}")
            # print(f"SSIM {_ssim}/{np.mean(ssim_ls)}")

            pbar.set_description(
                f"PSNR: {_psnr:.2f}/{np.mean(psnr_ls):.2f} | SSIM {_ssim:.2f}/{np.mean(ssim_ls):.2f}"
            )
            pbar.update(1)

    print(f"average PSNR is {np.mean(psnr_ls)}")
    print(f"average SSIM is {np.mean(ssim_ls)}")