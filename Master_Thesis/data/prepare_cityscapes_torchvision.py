import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F

import torchvision
import torchvision.transforms.functional as TF
# import torchvision.transforms as tvt
import torchvision.transforms.v2 as tvt
from torchvision import datasets as vision_datasets
from torchvision.datasets import Cityscapes
import datasets
from datasets import load_dataset

from PIL import Image
from PIL.Image import Resampling

import torch.distributed as dist

import numpy as np
import pandas as pd
import os
from typing import Any


# Loading Cityscapes from local directory and using torchvision
def load_cityscapes_tv(split="train", path="./datasets/cityscapes"):
    """
    Make sure to place gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip downloaded from official repo in path.
    """
    if not os.path.exists(path):
        raise ValueError(f"Could not find path {path}. Please check it...")
    print(f"[INFO]: Loading Cityscapes {split} Dataset from path: {path}...")
    dataset = Cityscapes(root=path,
                         split=split, mode="fine",
                         target_type="semantic")
    # we need to wrap when using v2 transforms on torchvision dataset
    dataset = vision_datasets.wrap_dataset_for_transforms_v2(dataset)
    return dataset


class CustomCityscapesDataset_tv(Dataset):
    def __init__(self, dataset, split="train",
                 num_classes=19, crop_size=768, ignore_index=255):  # 1024
        self.split = split
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.mask_transforms = RemapLabels(ignore_index=self.ignore_index, num_classes=self.num_classes)
        self.crop_size = crop_size
        self.dataset = dataset

    def get_train_transforms(self):
        transforms = tvt.Compose([
            # tvt.RandomCrop(size=self.crop_size, padding=True, fill=255, pad_if_needed=True, padding_mode="constant"),
            tvt.RandomResizedCrop(size=self.crop_size, scale=(0.5, 1.5), ratio=(0.75, 1.33), antialias=True),
            tvt.RandomHorizontalFlip(p=0.5),
            tvt.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # FIX ME: Normalize the mask is not needed rght? # use v2?
            tvt.ToTensor()])
        return transforms

    def get_fixed_transforms(self, split, sample):
        if split == "validation":
            fix_transforms = FixScaleCrop(crop_size=self.crop_size)
        elif split == "test":
            fix_transforms = FixedResize(size=self.crop_size)
        return fix_transforms(sample)

    def get_val_test_transforms(self):
        transforms = tvt.Compose([
            tvt.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tvt.ToTensor()
        ])
        return transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Any:
        image, mask = self.dataset[idx]["image"], self.dataset[idx]["mask"]

        # Remapping Labels first
        semantic_mask = self.mask_transforms(mask)

        # Transforms
        if self.split == "train":
            transforms = self.get_train_transforms()
            transformed_image = transforms(image)
            transformed_mask = transforms(semantic_mask)

        elif self.split in ("validation", "test"):
            sample = {"image": image, "mask": semantic_mask}
            # Fixed Transforms
            sample_transformed = self.get_fixed_transforms(self.split, sample)
            # Common Transforms
            common_transforms = self.get_val_test_transforms()
            transformed_image = common_transforms(sample_transformed["image"])
            transformed_mask = common_transforms(sample_transformed["mask"])

        return {
            "image": transformed_image,
            "mask": transformed_mask,
            # FIX ME: get the image name too
        }


def get_cityscapes_labels(num_classes=19):
    print("[INFO]: Getting the modified Cityscapes Labels")
    class_labels = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus", "train",
        "motorcycle", "bicycle"
    ]  # Here unlabelled is not considered, 255 is the index we use to ignore those unlabelled.
    
    assert len(class_labels) == num_classes, f"Existing Class labels are {len(class_labels)}, but num_classes given as {num_classes}."
    
    id2label = {idx: class_labels[idx] for idx in range(num_classes)}
    label2id = {class_labels[idx]: idx for idx in range(num_classes)}
    return id2label, label2id, class_labels


class RemapLabels(object):
    def __init__(self, ignore_index, num_classes):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.modify_labels()

    def modify_labels(self):
        print("[INFO]: Modifying the Cityscapes Labels")
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.id2label, self.label2id, self.class_labels = get_cityscapes_labels(self.num_classes)
        self.class_map = dict(zip(self.valid_classes, range(self.num_classes)))

    def __call__(self, mask):
        # Put all void classes to background
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
    def get_id2label(self):
        return self.id2label

    def get_label2id(self):
        return self.label2id

    def get_class_labels(self):
        return self.class_labels


class FixScaleCrop(object):
    def __init__(self, crop_size):
        """
        Fixed Scale Crop Transform for Validation dataset.
        """
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["mask"]
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
  
        # PIL Resize
        img = img.resize(size=(ow, oh), resample=Resampling.BILINEAR)
        mask = mask.resize(size=(ow, oh), resample=Resampling.NEAREST)

        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {"image": img,
                "mask": mask}


class FixedResize(object):
    """
    Fixed Resize Transform for Test dataset.
    """
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["mask"]

        assert img.size == mask.size

        img = img.resize(size=self.size, resample=Resampling.BILINEAR)
        mask = mask.resize(size=self.size, resample=Resampling.NEAREST)

        return {"image": img,
                "mask": mask}
