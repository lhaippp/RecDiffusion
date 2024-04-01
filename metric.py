import cv2
import glob
import numpy as np

from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


def psnrsk(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

results = glob.glob("DIR-D/testing/gt/*.jpg")

psnr_ls = []
ssim_ls = []

with tqdm(
    initial=0,
    total=len(results),
) as pbar:
    for res in results:
        img1 = cv2.imread(res)
        img2 = cv2.imread("CDM_Results/" + res.split('/')[-1].replace('.jpg', '.png'))

        _psnr = psnrsk(img1, img2)
        _ssim = ssim(img1, img2, channel_axis=2)

        psnr_ls.append(_psnr)
        ssim_ls.append(_ssim)

        pbar.set_description(
            f"PSNR: {_psnr:.2f}/{np.mean(psnr_ls):.2f} | SSIM {_ssim:.2f}/{np.mean(ssim_ls):.2f}"
        )
        pbar.update(1)

print(f"average PSNR is {np.mean(psnr_ls)}")
print(f"average SSIM is {np.mean(ssim_ls)}")
