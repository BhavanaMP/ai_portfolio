import torch
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

import numpy as np
from typing import Any


def load_streethazards(split="data", path="BhavanaMalla/streethazards"):
    dataset = load_dataset(path, split=split)
    return dataset


def get_streethazards_labels():
    """
    The ground truth contains labels for ID and OoD pixels, as well as ignored void pixels.
    """
    id2label = {
        1: "building",
        2: "fence",
        3: "other",
        4: "pedestrian",
        5: "pole",
        6: "road line",
        7: "road",
        8: "sidewalk",
        9: "vegetation",
        10: "car",
        11: "wall",
        12: "traffic sign",
        13: "anomaly"
    }
    label2id = {v: k for k, v in id2label.items()}
    labels = ["building", "fence", "other", "pedestrian", "pole", "road line", "road", "sidewalk", "vegetation", "car", "wall", "traffic sign", "anomaly"]
    return id2label, label2id, labels


def get_streethazards_test_transforms():
    # We use full size
    transforms = A.Compose([
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transforms


class CustomStreetHazardsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split="data", crop_size=1024, ignore_index=255):  # 768
        """
        Load streethazards dataset from Hugging Face.
              
        The ground truth contains labels for ID(0) and OoD(1) pixels, as well as ignored void(255) pixels.

        Existing Features:
        - 'image': Image(mode=None, decode=True, id=None), size: (720, 1280, 3)
        - 'mask': Image(mode=None, decode=True, id=None),
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
        semantic_mask = self.dataset[idx]["mask"]
        image_name = self.dataset[idx]["image_name"]

        # Transforms
        if self.split == "data":
            transforms = get_streethazards_test_transforms()
        else:
            raise ValueError(f"Wrong Split {self.split} given")

        transformed = transforms(image=np.array(image), mask=np.array(semantic_mask))
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return {"image": transformed_image, "mask": transformed_mask, "name": image_name}
