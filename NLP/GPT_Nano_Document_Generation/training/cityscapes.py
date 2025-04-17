import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F

import torchvision
import torchvision.transforms.functional as TF
# import torchvision.transforms as tvt
import torchvision.transforms.v2 as tvt
from torchvision import datasets
from torchvision.datasets import Cityscapes

from PIL import Image
from PIL.Image import Resampling
from tqdm.auto import tqdm

import torch.distributed as dist

import numpy as np
import pandas as pd
import os
from typing import Any
import argparse


def load_cityscapes(split="train", path="./datasets/cityscapes"):
    if not os.path.exists(path):
        raise ValueError(f"Could not find path {path}. Please check it...")
    print(f"[INFO]: Loading Cityscapes {split} Dataset from path: {path}...")
    dataset = Cityscapes(root=path,
                         split=split, mode="fine",
                         target_type="semantic")
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset)  # we need to wrap when using v2 transforms on torchvision dataset when not using any custom dataset
    return dataset


def _prepare_dataloader(dataset, batch_size, is_distributed, is_train):
    # Distributed sampler
    sampler = DistributedSampler(dataset) if is_distributed and is_train else None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True,
        drop_last=True,  # False
        shuffle=(is_train and sampler is None), sampler=sampler)
    return dataloader

class CustomCityscapesDataset(Dataset):
    def __init__(self, dataset, split="train",
                 num_classes=19, crop_size=768, ignore_index=255):  # 1024
        self.split = split
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.convert_pil = tvt.PILToTensor()
        self.mask_transforms = RemapLabels(ignore_index=self.ignore_index,
                                           num_classes=self.num_classes)
        self.crop_size = crop_size
        self.dataset = dataset

    def get_train_transforms(self):
        
        transforms = tvt.Compose([
            # tvt.RandomCrop(size=self.crop_size, padding=True, fill=255, pad_if_needed=True, padding_mode="constant"),
            tvt.RandomResizedCrop(size=self.crop_size, scale=(0.5, 1.5), ratio=(0.75, 1.33), antialias=True),
            tvt.RandomHorizontalFlip(p=0.5),
            tvt.ToDtype(torch.float32, scale=True),  # replaces ToTensor() thats gonna deprecate. Converts image to a float tensor scaled to [0, 1].
            tvt.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transforms

    def get_fixed_transforms(self, split, sample):
        # if split == "val":
        fix_transforms = FixScaleCrop(crop_size=self.crop_size)
        return fix_transforms(sample)

    def get_val_test_transforms(self):
        transforms = tvt.Compose([
            tvt.ToDtype(torch.float32, scale=True),
            tvt.Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
        ])
        return transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        image, mask = self.dataset[index]  # tuple(PIL.Image.Image, torchvision.tv_tensors._mask.Mask: torch.uint8)
        
        # Convert PIL Image to Tensor 
        image = self.convert_pil(image)
        
        # Remapping Labels first
        semantic_mask = self.mask_transforms(mask)

        # Transforms
        if self.split == "train":
            transforms = self.get_train_transforms()
            transformed_image, transformed_mask = transforms(image, semantic_mask)

            print(f"Train transformed_image.shape: {transformed_image.shape}, {type(transformed_image)}")  # Remove later
            print(f"Train transformed_mask.shape: {transformed_mask.shape}, {type(transformed_mask)}")  # Remove later
            print(f"Train Unique Mask values: {torch.unique(transformed_mask)}")  # Remove later

        elif self.split in ("val", "test"):
            sample = {"image": image, "mask": semantic_mask}
            # Fixed Transforms for just val, and test inference will be in original resolution
            if self.split == "val":
                sample = self.get_fixed_transforms(self.split, sample)
            # Common Transforms
            common_transforms = self.get_val_test_transforms()
            transformed_image, transformed_mask = common_transforms(sample["image"], sample["mask"])

            print(f"{self.split} transformed_image.shape: {transformed_image.shape}")  # Remove later
            print(f"{self.split} transformed_mask.shape: {transformed_mask.shape}")  # Remove later
            print(f"{self.split} Unique Mask values: {torch.unique(transformed_mask)}")  # Remove later

        return {
            "image": transformed_image,
            "mask": transformed_mask,
        }


class RemapLabels(object):
    def __init__(self, ignore_index, num_classes):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.modify_labels()

    def modify_labels(self):
        print("[INFO]: Modifying the Cityscapes Labels")
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_labels = [
            "unlabelled", "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
            "sky", "person", "rider", "car", "truck", "bus", "train",
            "motorcycle", "bicycle"
        ]
        assert len(self.valid_classes) == self.num_classes, f"Class labels are {len(self.class_labels)}, but num_classes given is {self.num_classes}."

        self.class_map = dict(zip(self.valid_classes, range(self.num_classes)))
        self.id2label = {idx: self.class_labels[idx] for idx in range(self.num_classes)}
        self.label2id = {self.class_labels[idx]: idx for idx in range(self.num_classes)}

    def __call__(self, mask):
        # Put all void classes to background
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


class FixScaleCrop(object):
    def __init__(self, crop_size):
        """
        Fixed Scale Crop Transform for Validation dataset.
        """
        self.crop_size = crop_size

    def __call__(self, sample):
        print("fixedscalecrop")
        img = sample["image"]
        mask = sample["mask"]
        _, h, w = img.shape
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        
        # Resize
        img = TF.resize(img, size=[oh, ow], interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size=[oh, ow], interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        # center crop
        _, h, w = img.shape
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = TF.crop(img, top=y1, left=x1, height=self.crop_size, width=self.crop_size)
        mask = TF.crop(mask, y1, x1, self.crop_size, self.crop_size)

        return {"image": img, "mask": mask}


class FixedResize(object):
    """
    Fixed Resize Transform for Test dataset.
    """
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        print("fixed resize")
        img = sample["image"]
        mask = sample["mask"]
        print(type(img), type(mask), img.size(), mask.size())
        img = TF.resize(img, self.size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        print(img.size(), mask.size())
        return {"image": img,
                "mask": mask}


def main(args):
    val_split = load_cityscapes(split="val", path=args.path)
    dataset = CustomCityscapesDataset(val_split, split="train")
    is_distributed = False
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        is_distributed = world_size >= 1
    # DataLoaders
    train_dataloader = _prepare_dataloader(dataset,
                                           batch_size=2,
                                           is_distributed=is_distributed,
                                           is_train=True)
    for step, batch in enumerate(tqdm(train_dataloader)):
        image = batch["image"]
        mask = batch["mask"]
        print(f"Step: {step} | Image: Type - {type(image)}, dtype: {image.dtype}, Shape - {image.shape}, Max - {torch.max(image)} , Min - {torch.min(image)}, Mean: {image[0].mean()}, Std: {image[0].std()} |  \nMask: Type - {type(mask)}, dtype: {mask.dtype}, Shape - {mask.shape}, Unique Labels - {torch.unique(mask)}")
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="D://cityscapes")
    args = parser.parse_args()
    # Call the main
    main(args)