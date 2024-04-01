from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision import transforms as T
from PIL import Image

from flow_field import convert_mesh_to_dxdy


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class DIRD_Dataset(Dataset):
    def __init__(
        self,
        root,
        image_size: Tuple[int, int],
        training_image_size: Tuple[int, int],
        enable_cache=False,
    ):
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.training_image_size = training_image_size
        self.files = [p.name for p in (self.root / "input").glob("*.jpg")]
        self.transform = T.ToTensor()
        self.cache = [None for _ in range(len(self.files))] if enable_cache else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]
        cond = self.transform(Image.open(self.root / "input" / self.files[index])).expand((3, -1, -1))
        gt = self.transform(Image.open(self.root / "gt" / self.files[index])).expand((3, -1, -1))
        mask = self.transform(Image.open(self.root / "mask" / self.files[index]).convert("L"))[:1]
        mesh = torch.tensor(np.load(self.root / "mesh" / self.files[index].replace("jpg", "npy")))
        flow = convert_mesh_to_dxdy(mesh, self.image_size)
        cond = torch.cat((cond, mask))
        assert cond.shape[0] == 4
        downsampled_cond = interpolate(cond.unsqueeze(0), size=self.training_image_size, mode="bilinear", align_corners=True).squeeze(0)
        downsampled_gt = interpolate(gt.unsqueeze(0), size=self.training_image_size, mode="bilinear", align_corners=True).squeeze(0)
        r = [flow, downsampled_cond, downsampled_gt, cond, gt, self.files[index].rstrip(".jpg") + ".png"]
        if self.cache is not None:
            self.cache[index] = r
        return r


if __name__ == "__main__":
    from flow_utils import flow_warp, upsample2d_flow_as
    from torchvision.utils import save_image

    dataset = DIRD_Dataset("../DIR-D/testing", (384, 512), (192, 256))
    flow, cond, gt, _, _, _ = dataset[0]
    flow, cond, gt = flow.unsqueeze(0), cond.unsqueeze(0), gt.unsqueeze(0)
    flow = upsample2d_flow_as(flow, cond, "bilinear", True)
    save_image(flow_warp(cond, flow).squeeze(0), "test.out.png")
    save_image(cond, "test.in.png")
