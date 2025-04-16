import torch
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

import numpy as np
from typing import Any


def load_lostandfound_full(split="test", path="BhavanaMalla/lostandfound-semantic"):
    dataset = load_dataset(path, split=split)
    return dataset


def get_lostandfound_full_labels():
    """
    The ground truth contains labels for ID and OoD pixels, as well as ignored void pixels.
    """
    id2label = {0: "id", 1: "ood", 2: "background"}
    label2id = {"id": 0, "ood": 1, "background": 2}
    labels = ["id", "ood", "background"]
    return id2label, label2id, labels


def get_lostandfound_full_train_transforms(crop_size=1024):  # 768
    transforms = A.Compose([
        A.LongestMaxSize(max_size=crop_size),  # Resizes the image so that the longest side is equal to crop_size, other side is resized for preserving the aspect ratio i.eotherside = new longest size / original asp ratio
        A.PadIfNeeded(min_height=crop_size, min_width=crop_size, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
        A.RandomResizedCrop(height=crop_size, width=crop_size, scale=(0.5, 1.5), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transforms


def get_lostandfound_full_test_transforms():
    # We use full size
    transforms = A.Compose([
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transforms


class CustomLostAndFoundFullDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split="test", crop_size=1024, ignore_index=255):  # 768
        """
        Load Lost and Found from dataset HF. This is combination of both train and test sets.
        
        Around 1036 train anf 1203 test images.
              
        The ground truth contains labels for ID(0) and OoD(1) pixels, as well as ignored void(1) pixels.
        

        Existing Features:
        - 'image' - size: (3, 1024, 2048)
        - 'annotation_color',
        - 'annotation_instanceids', 'annotation_labelids', 'annotation_labeltrainids', 'image_name'
        - Note: annotation_labeltrainids is the semantic annotation
        
        
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
            transforms = get_lostandfound_full_train_transforms()
        elif self.split == "test":
            transforms = get_lostandfound_full_test_transforms()
        else:
            raise ValueError(f"Wrong Split {self.split} given")

        transformed = transforms(image=np.array(image), mask=np.array(semantic_mask))
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return {"image": transformed_image, "mask": transformed_mask, "name": image_name}
