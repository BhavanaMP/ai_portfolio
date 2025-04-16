from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

# Distributed Training Imports DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import segmentation_models_pytorch as smp

from tqdm import tqdm

from config_utils import config_to_string
from modelling.model_utils import _get_train_datasets
from utils.common_utils import set_seeds


def prepare_model(encoder, decoder, input_channels):
    """
    Get the pretrained Model from segmentation_models_pytorch with the given encoder and decoder loaded with imagenet weights.
    This function also takes care of enabling dropout probabilities for encoders.
    """
    if decoder == "FPN":
        common_args = {"encoder_name": encoder, "encoder_depth": 5, "encoder_weights": "imagenet", "decoder_dropout": 0.3, "in_channels": 3, "classes": input_channels, "activation": None}
        model = smp.FPN(**common_args)
    elif decoder == "Unet":
        common_args = {"encoder_name": encoder, "encoder_depth": 5, "encoder_weights": "imagenet", "in_channels": 3, "classes": input_channels, "activation": None}
        model = smp.Unet(**common_args)
    elif decoder == "DeepLabV3Plus":
        common_args = {"encoder_name": encoder, "encoder_depth": 4, "encoder_weights": "imagenet", "classes": input_channels, "activation": None}
        model = smp.DeepLabV3Plus(**common_args)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(model)
    return model


def unnormalize_image(img, dataset_name):
    """
    Unnormalize the images before plotting
    """
    if dataset_name == "railsem19":
        img *= 255
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device).view(3, 1, 1)
        img = img * std + mean
    return img


def plot_reconstructed_images(val_data, writer, epoch, dataset_name):
    # Plot original and reconstructed images
    original_imgs, reconstructed_imgs = zip(*val_data)
    # Convert to tensors
    original_imgs = torch.stack(original_imgs).detach().cpu()
    reconstructed_imgs = torch.stack(reconstructed_imgs).detach().cpu()
    # Loop
    for i, (original_img, reconstructed_img) in enumerate(zip(original_imgs, reconstructed_imgs)):
        # Unnormalize
        original_img = unnormalize_image(original_img, dataset_name)
        reconstructed_img = unnormalize_image(reconstructed_img, dataset_name)
        try:
            # Prepare grid for visualization
            grid_image = vutils.make_grid([original_img, reconstructed_img], nrow=3, normalize=False)
            # Log to TensorBoard
            writer.add_image(f"SelectedIndices/Original and Reconstructed_{i}", grid_image, epoch)
        except Exception as e:
            print(f"Exception in creating grid image of Validation Predictions: {e}")


class AutoEncoder_SMP(nn.Module):
    """
    Monte Carlo Dropout  Deep Learning Classifier
    """
    def __init__(self, encoder, decoder, input_channels=3, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
    
        self.model = prepare_model(encoder=encoder, decoder=decoder,
                                   input_channels=self.input_channels)  # Set to num_classes to input_channels for autoencoder

        if isinstance(device, int):  # Check if device is an integer
            self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):  # Check if device is a string
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images):
        logits = self.model(images)
        return logits


class AutoEncoderPreTrainer:
    def __init__(self, args, gpu_id):
        self.seed = args.seed
        print(f"[INFO]: Setting the seed {self.seed}")
        set_seeds(self.seed)
        self.device = gpu_id

        self.is_train = args.is_pretrain
        self.dataset_name = args.train_dataset_name
        self.crop_size = args.crop_size
        self.ignore_index = args.ignore_index

        # Get the Dataset and labels
        self.train_dataset, self.val_dataset, _, _, _ = _get_train_datasets(self.dataset_name, self.ignore_index, self.crop_size)

        # Get the MCD Classifier
        self.model = AutoEncoder_SMP(encoder=args.encoder, decoder=args.decoder, input_channels=3, device=self.device)
        # Convert BatchNorm layers to SyncBatchNorm layers if present
        if self.has_batchnorm():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.batch_size = args.batch_size
        self.is_distributed = args.is_distributed
        self.epochs = args.num_epochs
        self.initial_lr = args.lr
        self.best_loss = 1e+5
        self.start_epoch = 0
        self.world_size = args.world_size
        self.model.to(self.device)
        # Optimizer
        self.optimizer = AdamW(params=self.model.model.parameters(), lr=self.initial_lr, weight_decay=1e-4)

        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", factor=0.5, patience=5, threshold=0.0001, min_lr=1e-6, verbose=True)
        # Scaler
        self.scaler = GradScaler()
      
        # Checkpoint saving path
        self.model_ckpt_path = Path(f"{args.pretrain_ckpt_save_loc}", f"model_{args.name}_{self.dataset_name}_{args.run_id[-7:]}.pt").absolute().__str__()
        
        # Model Results
        self.model_results = {}

        # Metrics and results
        self.metric_names = ["loss"]
        self.metric_types = ["train", "val"]

        # Loss
        # self.criterion = L1Loss().to(self.device)
        self.criterion = MSELoss().to(self.device)
        
        self._summary_writer = None
        # Get TensorBoard SummaryWriter
        self.log_dir = f"runs/{args.run_id}_{args.name}"
        os.makedirs(self.log_dir, exist_ok=True)

        # we initialize summary writer only in the main process
        if self.is_main_process():
            self._summary_writer = SummaryWriter(log_dir=self.log_dir, purge_step=None)
            config_str = config_to_string(args)
            self._summary_writer.add_text(tag="Configuration", text_string=config_str)

        self.log = self._summary_writer is not None

        # Wrapping model into DDP
        if self.is_distributed_mode():
            print("[INFO]: Wrapping the model into DDP")
            self.model = DDP(module=self.model, device_ids=[gpu_id])

    def has_batchnorm(self):
        """
        Checks if a model contains any batch normalization layers.
        """
        for layer in self.model.modules():
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return True
        return False

    def is_distributed_mode(self):
        return self.is_distributed and dist.is_initialized() and torch.cuda.is_available()
    
    def is_main_process(self):
        # To check if its the main rank or not distributed i.e CPU
        return not self.is_distributed_mode() or dist.get_rank() == 0

    def _prepare_dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int, is_distributed, is_train):
        # Distributed sampler
        sampler = DistributedSampler(dataset) if is_distributed and is_train else None
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=(is_train and sampler is None), sampler=sampler)
        return dataloader

    def _save_checkpoint(self):
        print("[INFO]: Saving checkpoint...")
        if hasattr(self.model, "module"):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        mld_ckpt = {
            "model_state_dict": model_state_dict
        }
        if os.path.exists(self.model_ckpt_path):
            print(f"[INFO]: Old Checkpoint File {self.model_ckpt_path} exists.. Removing it...")
            os.remove(self.model_ckpt_path)
        torch.save(obj=mld_ckpt, f=self.model_ckpt_path)
        print(f"[INFO]:Epoch {self.start_epoch} | Training checkpoint saved at {self.model_ckpt_path}.")

    def train_step(self, train_dataloader, epoch):
        running_train_loss = 0.0
        # Training Loop
        self.model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            self.optimizer.zero_grad()
            images = batch["image"].to(self.device)
            # Forward Pass
            with autocast():
                outputs = self.model(images)
                train_loss = self.criterion(outputs, images)
            print(f"\nTrain Step: {step}, Train Loss: {train_loss.item():.3f}")
            
            # Metrics calculation per batch
            running_train_loss += train_loss.item()
            # Backward pass
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            del batch

        # Epoch Metrics
        train_epoch_loss = running_train_loss / len(train_dataloader)
        if self.is_distributed_mode():
            train_epoch_loss = torch.tensor([train_epoch_loss], device=dist.get_rank())
            dist.all_reduce(train_epoch_loss, op=dist.ReduceOp.AVG)  # SUM(all gpus loss) / worldsize
        train_loss = train_epoch_loss.item()
        # Logging
        if self.log and self.is_main_process():
            self._summary_writer.add_scalar(tag=f"Train/Loss", scalar_value=train_loss, global_step=epoch)
        return train_loss

    def val_step(self, val_dataloader, epoch):
        running_val_loss = 0.0
        selected_data = []
        # Eval Loop
        self.model.eval()
        for step, batch in enumerate(tqdm(val_dataloader)):
            val_images = batch["image"].to(self.device)
            # Eval forward pass
            with torch.no_grad():
                val_outputs = self.model(val_images)
                val_loss = self.criterion(val_outputs, val_images)
            print(f"\nVal Step: {step}, Val Loss: {val_loss.item():.3f}")
            # Metrics calculation per batch
            running_val_loss += val_loss.item()
            if len(selected_data) <= 1:
                idx = 0  # get the first idx of batches per step
                selected_data.append((val_images[idx].squeeze(), val_outputs[idx].squeeze()))
            del batch

        # Epoch Metrics
        val_epoch_loss = running_val_loss / len(val_dataloader)
        if self.is_distributed_mode():
            val_epoch_loss = torch.tensor([val_epoch_loss], device=dist.get_rank())
            dist.all_reduce(val_epoch_loss, op=dist.ReduceOp.AVG)  # SUM(all gpus loss) / worldsize

        val_loss = val_epoch_loss.item()
        # Logging
        if self.log and self.is_main_process():
            self._summary_writer.add_scalar(tag=f"Val/Loss", scalar_value=val_loss, global_step=epoch)
            # Logging of images and predictions at the end of validation epoch
            print("[INFO]: Logging Validation Images")
            if selected_data:
                plot_reconstructed_images(selected_data, self._summary_writer, epoch, self.dataset_name)
        return val_loss

    def add_metric(self, metric_type, metric_name, value):
        key = f"{metric_type}_{metric_name}"
        self.model_results.setdefault(key, []).append(value)

    def train(self):
        print("[INFO]: Started Training the model...")
        print(f"[INFO]: Datasets length: {len(self.train_dataset)}, {len(self.val_dataset)}")

        # DataLoaders
        train_dataloader = self._prepare_dataloader(self.train_dataset, batch_size=self.batch_size, is_distributed=self.is_distributed_mode(), is_train=self.is_train)
        val_dataloader = self._prepare_dataloader(self.val_dataset, batch_size=self.batch_size, is_distributed=self.is_distributed_mode(), is_train=self.is_train)

        # Define model_results keys dynamically, if not resumed
        for metric_type in self.metric_types:
            for metric_name in self.metric_names:
                full_metric_name = f"{metric_type}_{metric_name}"
                self.model_results.setdefault(full_metric_name, [])

        print(f"\n-------[GPU: {self.device}] | Batch Size: {train_dataloader.batch_size} | Train Steps: {len(train_dataloader)}-------")

        # Training Loop
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            if self.is_distributed_mode():
                # Shuffling for distributed sampler
                train_dataloader.sampler.set_epoch(epoch)
            # Step
            train_loss = self.train_step(train_dataloader, self.start_epoch)
            val_loss = self.val_step(val_dataloader, self.start_epoch)
            self.scheduler.step(val_loss)
            # Saving per epoch results to model_results dict
            for metric_type in self.metric_types:
                model_loss = train_loss if metric_type == "train" else val_loss
                self.add_metric(metric_type, "loss", model_loss)
            if self.is_main_process():
                print(f"\nEpoch : {self.start_epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | \n")
            if self.is_distributed_mode():
                # Synchronize processes after each epoch
                dist.barrier()
            # Saving checkpoint of best model based on val loss
            if val_loss < self.best_loss and self.is_main_process():
                self.best_loss = val_loss
                self._save_checkpoint()
            # Logging of epoch metrics
            if self.log and self.is_main_process():
                self._summary_writer.add_scalar("epoch", epoch, self.start_epoch)
                self._summary_writer.add_scalar("model_learning_rate", self.optimizer.param_groups[0]["lr"], self.start_epoch)
            # Increment the start epoch
            self.start_epoch += 1

        # Final barrier to ensure all processes complete before logging final results
        if self.is_distributed_mode():
            dist.barrier()
        # Logging Per Class Bar charts at the end of Training
        is_final_epoch = self.log and self.start_epoch == self.epochs
        if is_final_epoch and self.is_main_process():
            print(f"[INFO]: Final Epoch: {is_final_epoch}")
            self._summary_writer.close()
        if self.is_distributed_mode():
            dist.barrier()
