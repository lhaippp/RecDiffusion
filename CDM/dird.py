from pathlib import Path
from torch import cat as catTensor, ones_like as onesLikeTensor
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision import transforms as T
from PIL import Image


class Dataset(TorchDataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=["jpg", "jpeg", "png", "tiff"],
        mask_weight=0,
        dis_weight=0,
        ones_weight=1,
        zeros_weight=0,
    ):
        super().__init__()
        self.is_train = folder.endswith("training")
        folder = Path(folder)
        self.image_size = image_size
        if self.image_size.__len__ is not None:
            self.image_size = tuple(self.image_size)
            assert len(self.image_size) == 2, "image_size must be an int or a list of length 2"
        else:
            assert isinstance(self.image_size, int), "image_size must be an int or a list of length 2"
        self.paths = [p for ext in exts for p in (folder / "gt").glob(f"*.{ext}")]
        self.input_root = folder / "input"
        self.mask_root = folder / "mask"
        self.light_root = folder / "light"
        self.wm_root = folder / "wm"
        self.ref_root = folder / "output"
        self.transform = T.Compose([T.ToTensor()])

        self.mask_weight = mask_weight
        self.dis_weight = dis_weight
        self.ones_weight = ones_weight
        self.zeros_weight = zeros_weight

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        input = self.input_root / path.name
        mask = self.mask_root / path.name
        if not self.is_train:
            ref = self.ref_root / (path.name.rstrip(".jpg") + ".png")
            light = self.light_root / (path.name.rstrip(".jpg") + ".png")
            wm = self.wm_root / (path.name.rstrip(".jpg") + ".png")
        gt = self.transform(Image.open(path).convert("RGB")) * 2 - 1
        input = self.transform(Image.open(input).convert("RGB")) * 2 - 1
        mask = self.transform(Image.open(mask).convert("L"))
        input = catTensor((input, mask))
        ret = {"image": gt, "LR_image": input, "original_name": str(path.name).rstrip(".jpg")}

        if not self.is_train:
            wm = self.transform(Image.open(wm).convert("L"))
            ref = self.transform(Image.open(ref).convert("RGB")) * 2 - 1
            light = self.transform(Image.open(light).convert("L"))

            # mix weight
            weight = 1.0 - (light * self.dis_weight + (1.0 - mask) * self.mask_weight + onesLikeTensor(light) * self.ones_weight) / (
                self.dis_weight + self.mask_weight + self.ones_weight + self.zeros_weight
            )
            weight[wm > 0] = 0
            
            ret.update({"weight": weight, "reference": ref})

        return ret
