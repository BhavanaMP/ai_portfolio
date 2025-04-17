import torch
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

import numpy as np
from typing import Any

import torch.utils.data


def load_fishyscapes(split="data", path="BhavanaMalla/fishyscapes-semantic"):
    dataset = load_dataset(path, split=split)
    return dataset


def get_fishyscapes_labels():
    """
    The ground truth contains labels for ID and OoD pixels, as well as ignored void pixels.
    """
    id2label = {0: "id", 1: "ood", 255: "background"}
    label2id = {"id": 0, "ood": 1, "background": 255}
    labels = ["id", "ood", "background"]
    return id2label, label2id, labels


def get_fishyscapes_test_transforms():
    # We use full size
    transforms = A.Compose([
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transforms


class CustomFishyscapesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split="data", crop_size=1024, ignore_index=255):  # 768
        """
        Load Fishyscapes dataset from Hugging Face.

        The ground truth contains labels for ID(0) and OoD(1) pixels, as well as ignored void(255) pixels.

        There are 3 different labels in the Fishyscapes Lost & Found dataset:

            - anomaly (any anomalous object)
            - in-distribution (any object or background class that fits into one of the definitions of the cityscapes classes)
            - void (does not exactly fit to cityscapes classes, but could still be found in cityscapes training images, most often labelled also as void there).
        The images were labelled to strictly satisfy the definitions of the first two classes. Applying a coarse labelling that means that some pixels that are labelled void can also belong to the first two classes. In evaluation, we ignore all pixels that are labelled as void.

        Existing Features:
        - 'image': Image(mode=None, decode=True, id=None), # Size: (1024, 2048, 3)
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
            transforms = get_fishyscapes_test_transforms()
        else:
            raise ValueError(f"Wrong Split {self.split} given")

        transformed = transforms(image=np.array(image), mask=np.array(semantic_mask))
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return {"image": transformed_image, "mask": transformed_mask, "name": image_name}
