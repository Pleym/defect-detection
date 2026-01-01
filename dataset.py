import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path    


class MVtecDataset(Dataset):
    "
    -- good : 0
    -- defect : 1
    "

    def __init__(self,root_dir,split,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.samples = []
        self.load_samples()


    def load_samples(self):
        split_dir = self.root_dir / self.split
        if self.split = "train":
            good_dir = split_dir / "good"if self.root_dir / "test" :
if self.root_dir / "test" :

            self._add_images_from_dir(good_dir,label=0)
        else:
            for defect_type in split_dir.iterdir():
                if defect_type.name =="good":
                    self._add_images_from_dir(defect_type,label=0)
                else:
                    self._add_images_from_dir(defect_type,label=1)

    def _add_images_from_dir(self,directory,label):
        if not directory.exists():
            return
        for img_path in directory.glob("*.png"):
            self.samples.append((img_path, label))
    def __getitem__(self,idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label,dtype=torch.long)



