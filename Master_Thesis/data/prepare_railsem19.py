from pathlib import Path
import json
from copy import deepcopy
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as tvt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms.functional import crop
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def load_railsem_dataset():
    print("[INFO]: Extracting Railsem19 dataset from hugginface hub")
    railsem_ds = load_dataset("BhavanaMalla/railsem19-semantic-expanded")
    return railsem_ds


def load_railsem19_test_dataset(path: str = "BhavanaMalla/railsem19_test_split_original"):
    print(f"[INFO]: Extracting Railsem19 Test dataset from hf {path}")
    test_ds = load_dataset(path)
    return test_ds


def get_railsem19_labels():
    modified_id2label = {
        0: "road",
        1: "sidewalk",
        2: "construction_fence",
        3: "rail_raised_rail_embedded",
        4: "pole_traffic_light_traffic_sign",
        5: "sky",
        6: "human",
        7: "tram_track_rail_track",
        8: "car_truck",
        9: "on_rails",
        10: "vegetation",
        11: "trackbed",
        12: "background_terrain"
    }
    modified_label2id = {label: id for id, label in modified_id2label.items()}
    modified_labels = [label for label in modified_id2label.values()]
    return modified_id2label, modified_label2id, modified_labels


def load_railsem19_splits():
    print("[INFO]: Loading splits")
    data_directory = Path("./datasets/railsem19")
    csv_file_path = data_directory / "RAIL_SEM19_split.csv"
    if csv_file_path.is_file():
        print(f"[INFO]: Found {csv_file_path}.Skipping Download...")
    else:
        print("[INFO]: Downloading RAIL_SEM19_split.csv from hub")
        csv_file_path = hf_hub_download(repo_id="BhavanaMalla/railsem19-semantic-expanded", filename="RAIL_SEM19_split.csv",
                                        repo_type="dataset", local_dir=data_directory)
    splits_df = pd.read_csv(csv_file_path)
    # Filter rows based on type
    train_names = splits_df[splits_df["type"] == "Train"]["Names"].tolist()
    val_names = splits_df[splits_df["type"] == "Validation"]["Names"].tolist()
    test_names = splits_df[splits_df["type"] == "Test"]["Names"].tolist()
    return train_names, val_names, test_names


def get_railsem19_transforms():
    # Transforms
    image_transforms = tvt.Compose([tvt.ToTensor(),])  # ToTensor scales between 0 and 1, if we use normalize, the values goes beyond 0 and 1, the range could be outside of [0, 1] and would ideally have a zero mean and unit variance.
    mask_transforms = tvt.Compose([tvt.PILToTensor(),  # Doesnt scale, Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W). doesnot scale hence using this for masks
                                  RemapBackground(),
                                  RemapLabels(),])
    return image_transforms, mask_transforms


class CustomRailsem19Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms, split=None, ignore_index=255, crop_size=1024):
        """
        Load Railsem19 dataset from HF.
        
        Features:
        - image: size: (3, 1080, 1920)
        - semantic_mask_label
        - img_Name
        """
        self.dataset = dataset
        self.transforms = transforms
        self.split = split
        self.ignore_index = ignore_index
        self.crop_size = crop_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        semantic_mask = self.dataset[idx]["semantic_mask_label"]
        image_name = self.dataset[idx]["img_Name"]
        # Transforms
        image_transforms, mask_transforms = self.transforms
        transformed_image = image_transforms(image)
        transformed_mask = mask_transforms(semantic_mask)
        # Padding
        if not self.split == "train":
            # Padding check for val and test splits
            if (transformed_image.shape[1] % 32 != 0 or transformed_image.shape[2] % 32 != 0):
                height, width = transformed_image.shape[1], transformed_image.shape[2]
                # Calculate padding to ensure both height and width are divisible by 32
                pad_height = max(32 - height % 32, 0)
                pad_width = max(32 - width % 32, 0)
                pad_height_half = pad_height // 2
                pad_width_half = pad_width // 2
                # Calculate border for padding
                border = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
                # Pad the images
                transformed_image = F.pad(input=transformed_image, pad=border, mode="constant", value=0)
                transformed_mask = F.pad(input=transformed_mask, pad=border, mode="constant", value=self.ignore_index)
        if self.split == "train":
            random_scaler = RandResize(scale=(0.5, 2.0))
            transformed_image, transformed_mask = random_scaler(transformed_image.unsqueeze(0).float(), transformed_mask.unsqueeze(0).float())
            # Pad image if it's too small after the random resize
            if transformed_image.shape[1] < 1024 or transformed_image.shape[2] < 1024:
                height, width = transformed_image.shape[1], transformed_image.shape[2]
                pad_height = max(1024 - height, 0)
                pad_width = max(1024 - width, 0)
                pad_height_half = pad_height // 2
                pad_width_half = pad_width // 2
                border = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
                transformed_image = F.pad(input=transformed_image, pad=border, mode="constant", value=0)
                transformed_mask = F.pad(input=transformed_mask, pad=border, mode="constant", value=self.ignore_index)
            # Applying horizontal flip with 50% probability
            if random.random() < 0.5:
                transformed_image = TF.hflip(transformed_image)
                transformed_mask = TF.hflip(transformed_mask)
            # Random Crop
            i, j, h, w = tvt.RandomCrop(size=(1024, 1024)).get_params(transformed_image, output_size=(1024, 1024))
            transformed_image = TF.crop(transformed_image, i, j, h, w)
            transformed_mask = TF.crop(transformed_mask, i, j, h, w)
        return {"image": transformed_image, "mask": transformed_mask, "name": image_name}


class RandResize(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    Source: https://github.com/Haochen-Wang409/U2PL/blob/main/u2pl/dataset/augmentation.py
    """
    def __init__(self, scale, aspect_ratio=None):
        self.scale = scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, image, label):
        if random.random() < 0.5:
            temp_scale = self.scale[0] + (1.0 - self.scale[0]) * random.random()
        else:
            temp_scale = 1.0 + (self.scale[1] - 1.0) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = (
                self.aspect_ratio[0]
                + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            )
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        h, w = image.size()[-2:]
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)
        image = F.interpolate(
            image, size=(new_h, new_w), mode="bilinear", align_corners=False
        )  # bilinear
        label = F.interpolate(label, size=(new_h, new_w), mode="nearest")
        return image.squeeze(), label.squeeze(0).to(dtype=torch.int64)


class RemapBackground():
    """ Remap background to label 19 """
    def __call__(self, mask):
        return torch.where(mask > 18, 19, mask)


class RemapLabels():
    """
    Remap Original Labels to Modified Mapping
    """
    def __init__(self):
        self.class_mapping = self.get_railsem19_id2label_from_hf()
        self.class_coding = {v: k for k, v in self.class_mapping.items()}
        # Remapping of original 20 labels to 13 labels
        self.modified_labels = {
            0: ["road"],
            1: ["sidewalk"],
            2: ["construction", "fence"],
            3: ["rail_raised", "rail_embedded"],
            4: ["pole", "traffic_light", "traffic_sign"],
            5: ["sky"],
            6: ["human"],
            7: ["tram_track", "rail_track"],
            8: ["car", "truck"],
            9: ["on_rails"],
            10: ["vegetation"],
            11: ["trackbed"],
            12: ["background", "terrain"]
        }
        self.modified_ids = {}
        for k, v in self.modified_labels.items():
            self.modified_ids[k] = [self.class_coding[label] for label in v]

    def get_railsem19_id2label_from_hf(self):
        data_directory = Path("./datasets/railsem19")
        json_file_path = data_directory / "labels_info.json"
        if json_file_path.is_file():
            print(f"[INFO]: Found {json_file_path}.Skipping Download...")
        else:
            print("[INFO]: Downloading labels_info.json from hub")
            json_file_path = hf_hub_download(repo_id="BhavanaMalla/railsem19-semantic-expanded", filename="labels_info.json", repo_type="dataset", local_dir=data_directory)
        with open(json_file_path, "r") as f:
            labels_info = json.load(f)
        id2label = labels_info["id2label"]
        # add the background label
        id2label["19"] = "background"
        # correcting the labels
        id2label = {int(key): value.replace('-', '_') for key, value in id2label.items()}
        return id2label

    def __call__(self, mask):
        final_label = np.zeros_like(mask)
        for key, val in self.modified_ids.items():
            specific_label = np.zeros_like(mask)
            specific_label = np.where(np.isin(mask, np.array(val)), 1, 0)
            specific_label *= key
            final_label = np.add(final_label, specific_label)
        return torch.from_numpy(final_label)
