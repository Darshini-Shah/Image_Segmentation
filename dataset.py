import os
import random

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HorseDataset(Dataset):
    def __init__(self, image_files, mask_files, image_transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_transform = image_transform

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        mask_path = self.mask_files[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask


def get_image_transform():
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    return image_transform


def create_dataset_splits(image_dir, mask_dir, seed=42):
    image_names = sorted(os.listdir(image_dir))
    mask_names = sorted(os.listdir(mask_dir))

    image_files = []
    mask_files = []

    for image_name, mask_name in zip(image_names, mask_names):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)

        image_files.append(image_path)
        mask_files.append(mask_path)

    combined = list(zip(image_files, mask_files))

    random.seed(seed)
    random.shuffle(combined)

    image_files = []
    mask_files = []

    for image_path, mask_path in combined:
        image_files.append(image_path)
        mask_files.append(mask_path)

    total = len(image_files)

    train_end = int(0.70 * total)
    val_end = int(0.85 * total)

    train_images = image_files[:train_end]
    train_masks = mask_files[:train_end]

    val_images = image_files[train_end:val_end]
    val_masks = mask_files[train_end:val_end]

    test_images = image_files[val_end:]
    test_masks = mask_files[val_end:]

    return (
        train_images,
        train_masks,
        val_images,
        val_masks,
        test_images,
        test_masks
    )