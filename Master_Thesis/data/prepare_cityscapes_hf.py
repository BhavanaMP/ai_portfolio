from typing import Any

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2


def load_cityscapes(split="train", path="BhavanaMalla/cityscapes-instance-semantic"):
    print(f"[INFO]: Extracting the Cityscapes {split} Dataset from hugginface hub")  # 5000 with (1024 × 2048) , 19 classes # 2975, 500, 1525
    dataset = load_dataset(path, split=split)
    return dataset


def get_cityscapes_labels(num_classes=19):
    print("[INFO]: Getting the modified Cityscapes Labels")
    class_labels = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus", "train",
        "motorcycle", "bicycle"
    ]
    assert len(class_labels) == num_classes, f"Existing Class labels are {len(class_labels)}, but num_classes given as {num_classes}."
    id2label = {idx: class_labels[idx] for idx in range(num_classes)}
    label2id = {class_labels[idx]: idx for idx in range(num_classes)}
    return id2label, label2id, class_labels


def get_cityscapes_train_transforms(crop_size=1024, ignore_index=255):  # 768
    transforms = A.Compose([
        # A.LongestMaxSize(max_size=crop_size),  # Resizes the image so that the longest side is equal to crop_size, other side is resized for preserving the aspect ratio i.eotherside = new longest size / original asp ratio
        A.RandomResizedCrop(height=crop_size, width=crop_size, scale=(0.5, 1.5), ratio=(0.75, 1.33), p=1),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=ignore_index),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),  # Random color jitter
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transforms


def get_cityscapes_val_transforms(crop_size=1024, ignore_index=255):
    # Fixed crop transforms for val dataset
    transforms = A.Compose([
        A.Resize(height=crop_size, width=crop_size),  # Ensure the image is at least the crop size while preserving the aspect ratio. Needed, validation has small images (512, 1024)
        A.LongestMaxSize(max_size=crop_size),  # Resizes the image so that the longest side is equal to crop_size, other side is resized for preserving the aspect ratio i.eotherside = new longest size / original asp ratio
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=ignore_index),
        A.CenterCrop(height=crop_size, width=crop_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transforms


def get_cityscapes_test_transforms(ignore_index=255):
    # We use full size
    transforms = A.Compose([
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=ignore_index),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transforms


class CustomCityscapesDataset(Dataset):
    def __init__(self, dataset, split="train", crop_size=1024, ignore_index=255):  # 768
        """
        Load Cityscapes from HF dataset.

        Note: No need of remapping the labels.
              Its already done while preparing the hf dataset.
   
        Useful keys for semantic segmentation are: image, annotation_labeltrainids, image_name. 

        Existing Features:
        - 'image': Image(mode=None, decode=True, id=None), size: (1024, 2048, 3)
        - 'annotation_color': Image(mode=None, decode=True, id=None),
        - 'annotation_instanceids': Image(mode=None, decode=True, id=None),
        - 'annotation_labelids': Image(mode=None, decode=True, id=None),
        - 'annotation_labeltrainids': Image(mode=None, decode=True, id=None),
        - 'image_name': Value(dtype='string', id=None)}
        """
        self.split = split
        self.ignore_index = ignore_index
        self.crop_size = crop_size
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Any:
        image = self.dataset[idx]["image"]
        semantic_mask = self.dataset[idx]["annotation_labeltrainids"]
        image_name = self.dataset[idx]["image_name"]

        # Transforms
        if self.split == "train":
            transforms = get_cityscapes_train_transforms(self.crop_size, self.ignore_index)
        elif self.split == "validation":
            transforms = get_cityscapes_val_transforms(self.crop_size, self.ignore_index)
        elif self.split == "test":
            transforms = get_cityscapes_test_transforms(self.ignore_index)
        else:
            raise ValueError(f"Wrong Split {self.split} given")

        transformed = transforms(image=np.array(image), mask=np.array(semantic_mask))
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        original_height, original_width = np.array(image).shape[:2]

        return {"image": transformed_image, "mask": transformed_mask, "name": image_name,
                "original_height": original_height, "original_width": original_width}
