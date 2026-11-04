"""MVTec AD dataset and Lightning DataModule.

Expected layout: <root>/<category>/{train,test}/<defect>/*.png, where <defect> is "good"
for normal images. Train/val only use train/good; test mixes good and defective images.
"""
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def list_samples(root, categories, split):
    """Return (path, category, defect) for every image of the given split."""
    samples = []
    for category in categories:
        split_dir = Path(root) / category / split
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in sorted(defect_dir.glob("*.png")):
                samples.append((path, category, defect_dir.name))
    return samples


def load_image(path, img_size):
    """Load as RGB (screw is grayscale), resize and return a uint8 tensor (3, H, W)."""
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)


# Resized images kept per process: in a Hydra multirun the 8 jobs share one process, so only the first loads from disk
_IMAGE_CACHE = {}


def load_images(paths, img_size):
    """Stack of uint8 images (N, 3, H, W) for the given paths, cached in memory."""
    key = (tuple(str(p) for p in paths), img_size)
    if key not in _IMAGE_CACHE:
        _IMAGE_CACHE[key] = torch.stack([load_image(path, img_size) for path in paths])
    return _IMAGE_CACHE[key]


class MVTecDataset(Dataset):
    """Images are resized once and kept in memory (~50 KB each at 128x128), so epochs are fast."""

    def __init__(self, samples, img_size):
        self.categories = [category for _, category, _ in samples]
        self.defects = [defect for _, _, defect in samples]
        self.labels = [int(defect != "good") for defect in self.defects]  # 0 = good, 1 = defective
        self.images = load_images([path for path, _, _ in samples], img_size)

    def __len__(self):
        return len(self.images)

    def get_images(self, indices):
        """Float images in [0, 1] for the given indices."""
        return self.images[indices].float() / 255.0

    def __getitem__(self, idx):
        return {
            "image": self.images[idx].float() / 255.0,
            "label": self.labels[idx],
            "category": self.categories[idx],
            "defect": self.defects[idx],
        }


class MVTecDataModule(L.LightningDataModule):
    def __init__(self, root, classes, img_size=128, batch_size=32, val_split=0.15, num_workers=2, seed=42):
        super().__init__()
        self.root = root
        self.classes = list(classes)
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def setup(self, stage=None):
        if stage in ("fit", "validate", None) and self.train_dataset is None:
            train_samples, val_samples = self._split(list_samples(self.root, self.classes, "train"))
            self.train_dataset = MVTecDataset(train_samples, self.img_size)
            self.val_dataset = MVTecDataset(val_samples, self.img_size)
        if stage in ("test", None) and self.test_dataset is None:
            self.test_dataset = MVTecDataset(list_samples(self.root, self.classes, "test"), self.img_size)

    def _split(self, samples):
        """Hold out val_split of each category, so validation contains all 4 objects."""
        rng = random.Random(self.seed)
        train, val = [], []
        for category in self.classes:
            items = [s for s in samples if s[1] == category]
            rng.shuffle(items)
            n_val = int(round(len(items) * self.val_split))
            val += items[:n_val]
            train += items[n_val:]
        return train, val

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
