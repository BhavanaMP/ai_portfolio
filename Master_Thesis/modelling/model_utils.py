import os
import re

import torch
import torch.distributed as dist
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck

from data.prepare_railsem19 import CustomRailsem19Dataset, load_railsem_dataset, get_railsem19_transforms, get_railsem19_labels, load_railsem19_splits
from data.prepare_cityscapes_hf import CustomCityscapesDataset, load_cityscapes, get_cityscapes_labels

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders.resnet import pretrained_settings

from modelling.resnet_dropout import ResNetEncoderDropout
from utils.common_utils import print_trainable_parameters


def _get_train_datasets(dataset_name, ignore_index, crop_size):
    """
    Training only on railsem19 or Cityscapes.
    """
    if dataset_name == "cityscapes":
        train_split = load_cityscapes(split="train")
        val_split = load_cityscapes(split="validation")
        # train_split = train_split.select(indices=(range(80)))
        # val_split = val_split.select(indices=(range(40)))
        id2label, label2id, labels = get_cityscapes_labels()
        # Custom Dataset
        train_dataset = CustomCityscapesDataset(train_split, split="train", crop_size=crop_size, ignore_index=ignore_index)
        val_dataset = CustomCityscapesDataset(val_split, split="validation", crop_size=crop_size, ignore_index=ignore_index)
    elif dataset_name == "railsem19":
        railsem_ds = load_railsem_dataset()
        dataset = railsem_ds["data"]
        # Get the splits
        train_names, val_names, _ = load_railsem19_splits()
        train_idxs, val_idxs = [], []
        dataset.map(lambda img, idx: (train_idxs.append(idx) if img["img_Name"] in train_names else val_idxs.append(idx) if img["img_Name"] in val_names else None), with_indices=True)
        train_split = dataset.select(train_idxs)
        val_split = dataset.select(val_idxs)
        # train_split = train_split.select(indices=(range(80)))
        # val_split = val_split.select(indices=(range(40)))
        id2label, label2id, labels = get_railsem19_labels()
        # Get the transforms
        transforms = get_railsem19_transforms()
        # Custom Dataset
        train_dataset = CustomRailsem19Dataset(train_split, transforms, split="train", ignore_index=ignore_index, crop_size=crop_size)
        val_dataset = CustomRailsem19Dataset(val_split, transforms, split="val", ignore_index=ignore_index, crop_size=crop_size)
    else:
        raise ValueError(f"Training is only supported on 'cityscapes' or 'railsem19' dataset but dataset_name is given as {dataset_name} ")
    print(f"[INFO]: Total Training images: {len(train_split)}")
    print(f"[INFO]: Total Validation images: {len(val_split)}")
    return train_dataset, val_dataset, id2label, label2id, labels


def apply_resnet101_dropout(model):
    """
    Enable dropout for resnet101 encoder layers.
    """
    print("Enabling dropout for 'Resnet101' encoder")
    params = {"out_channels": (3, 64, 256, 512, 1024, 2048), "block": Bottleneck, "layers": [3, 4, 23, 3], "depth": 5}
    model.encoder = ResNetEncoderDropout(dropout_prob=0.5, **params)
    model.encoder.set_in_channels(3)
    model.encoder.load_state_dict(model_zoo.load_url(pretrained_settings["resnet101"]["imagenet"]["url"]))
    return model


def apply_mitb3_dropout(model):
    """
    Enable dropout for mit_b3 encoder layers.
    """
    print("Enabling dropout for 'mit_b3' encoder")
    for name, module in model.encoder.named_children():
        if name.startswith("block"):
            # Get the block number
            block_num = int(name.replace("block", ""))  
            if block_num == 3 or block_num == 4:
                # Iterating through modulelist of block
                for idx, block in enumerate(module):
                    # Apply dropout to last 7 layers of block 3
                    if block_num == 3 and idx >= len(module) - 10: 
                        for _, submodule in block.named_modules():
                            if isinstance(submodule, torch.nn.Dropout):
                                submodule.p = 0.5
                    elif block_num == 4:  # Apply dropout to entire block 4
                        for _, submodule in block.named_modules():
                            if isinstance(submodule, torch.nn.Dropout):
                                submodule.p = 0.4
    return model


def apply_convnext_dropout(model):
    """
    Enable dropout for tu-convnext_small encoder layers.
    """
    print("Enabling dropout for 'tu-convnext_small' encoder")
    for name, module in model.encoder.named_modules():
        if isinstance(module, torch.nn.Dropout):
            if name.startswith("model.stages_2."):
                pattern = re.compile(r'blocks\.(\d+)')
                stage2_numbers = int(pattern.search(name).group(1))
                if stage2_numbers > 18 and "drop" in str(name):
                    module.p = 0.5
            elif name.startswith("model.stages_3.") and "drop1" in str(name):
                module.p = 0.5
    return model


def get_smp_model(encoder, decoder, num_classes):
    """
    Get the pretrained Model from segmentation_models_pytorch with the given encoder and decoder loaded with imagenet weights.
    This function also takes care of enabling dropout probabilities for encoders.
    """
    if decoder == "FPN":
        common_args = {"encoder_name": encoder, "encoder_depth": 5, "encoder_weights": "imagenet", "decoder_dropout": 0.3, "in_channels": 3, "classes": num_classes, "activation": None}
        model = smp.FPN(**common_args)
    elif decoder == "Unet":
        common_args = {"encoder_name": encoder, "encoder_depth": 5, "encoder_weights": "imagenet", "in_channels": 3, "classes": num_classes, "activation": None}
        model = smp.Unet(**common_args)
    elif decoder == "DeepLabV3Plus":
        common_args = {"encoder_name": encoder, "encoder_depth": 4, "encoder_weights": "imagenet", "classes": num_classes, "activation": None}
        model = smp.DeepLabV3Plus(**common_args)
    # Enable dropout for encoders
    if encoder == "resnet101":
        model = apply_resnet101_dropout(model)
    elif "mit_b3" in encoder:
        model = apply_mitb3_dropout(model)
    elif "tu-convnext_small" in encoder:
        model = apply_convnext_dropout(model)  # later make it single function
    print_trainable_parameters(model, print_msg=(f"Model with {encoder} Encoder and {decoder} Decoder Trainable Params"))
    if not dist.is_initialized() or dist.get_rank() == 0:
        # Print if its cpu or rank 0
        print(model)
    return model


def resume_from_checkpoint(model, optimizer, scheduler, scaler, mdl_save_path):
    """
    This function allows to Resume the training by loading the model from the previously saved checkpoint.
    It loads the statedicts of previously saved checkpoint in events of training crash.
    """
    start_epoch = 0
    best_loss = 1e+5
    if os.path.exists(mdl_save_path):
        print("[INFO]: Loading the previously saved checkpoint from : {mdl_save_path}")
        loaded_ckpt = torch.load(f=mdl_save_path)
        # Loading the save state dicts
        model.load_state_dict(loaded_ckpt["model_state_dict"])
        optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(loaded_ckpt["scheduler_state_dict"])
        scaler.load_state_dict(loaded_ckpt["scaler_state_dict"])
        best_loss = loaded_ckpt["best_loss"]
        start_epoch = loaded_ckpt["epoch"] + 1
        return {"model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler, "bets_loss": best_loss, "start_epoch": start_epoch}
    else:
        raise FileNotFoundError(f"Failed Resuming. {mdl_save_path} Doesnt not exist.")


def enable_dropout(model):
    """ Function to enable the dropout layers during evaluation."""
    print("[INFO]: Enabling dropout to activate MC dropout")
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    return model


def move_to_cpu(value):
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    elif isinstance(value, torch.Tensor):
        if value.device.type == 'cuda':
            if value.requires_grad:
                return value.detach().cpu()
            else:
                return value.cpu()
    return value


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
