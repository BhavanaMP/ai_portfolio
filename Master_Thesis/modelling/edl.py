from pathlib import Path
import os
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Distributed Training Imports DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, JaccardIndex

from config_utils import config_to_string
from modelling.poly_lr import PolyLR
from modelling.edl_classifier import EDLClassifier
from modelling.model_utils import _get_train_datasets, resume_from_checkpoint, move_to_cpu
from losses.edl_loss import EDLLoss
from utils.common_utils import set_seeds
from utils.plot_utils import plot_validation_predictions


class EDLTrainer:
    def __init__(self, args, gpu_id):
        self.seed = args.seed
        print(f"[INFO]: Setting the seed {self.seed}")
        set_seeds(self.seed)

        # Add Hparams
        self.hparams = {
            "encoder": args.encoder, "decoder": args.decoder, "num_epochs": args.num_epochs, "lr": args.lr, "batch_size": args.batch_size,
            "name": args.name, "model_type": args.model_type, "dataset_name": args.train_dataset_name, "run_id": args.run_id,
            "edl": args.train_edl, "ohem": args.use_ohem,
            "crop_size": args.crop_size, "ignore_index": args.ignore_index, "resumed": args.resume_training
        }

        self.is_train = args.is_train
        self.dataset_name = args.train_dataset_name
        self.crop_size = args.crop_size
        self.ignore_index = args.ignore_index
        self.train_edl = args.train_edl
        self.use_uncertainty_in_edl = args.use_uncertainty_in_edl
        self.use_ohem = args.use_ohem

        # Get the Dataset and labels
        self.train_dataset, self.val_dataset, self.id2label, self.label2id, self.labels = _get_train_datasets(self.dataset_name, self.ignore_index, self.crop_size)
        self.num_classes = len(self.id2label)

        self.batch_size = args.batch_size
        self.epochs = args.num_epochs
        self.initial_lr = args.lr
        self.is_distributed = args.is_distributed
        self.device = gpu_id
        self.world_size = args.world_size
        self.best_loss = 1e+5
        self.best_mIoU = 0 
        self.start_epoch = 0

        # Now get the EDL classifier
        self.model = EDLClassifier(num_classes=self.num_classes, encoder=args.encoder, decoder=args.decoder)
        # Convert BatchNorm layers to SyncBatchNorm layers if present
        if self.has_batchnorm():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        # Finetune related args
        self.is_finetune = args.is_finetune
        if self.is_finetune:
            self.pretrained_mdl_ckpt_path = args.pretrained_mdl_ckpt_path
            if not os.path.exists(self.pretrained_mdl_ckpt_path):
                raise ValueError(f"Pretrained checkpoint path: {self.pretrained_mdl_ckpt_path} not found...")
            # Load the pretrained weights
            pretrained_state_dict = torch.load(self.pretrained_mdl_ckpt_path)
            # Ensure only layers that exist in both models are transferred
            model_dict = self.model.state_dict()
            # pretrained_state_dict = {k: v for k, v in pretrained_state_dict["model_state_dict"].items() if k in model_dict and not "segmentation_head" in k}
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict["model_state_dict"].items() if k in model_dict}
            # Load the pretrained state_dict into the new model
            model_dict.update(pretrained_state_dict)
            self.model.load_state_dict(model_dict)
            # # Freeze the encoder layers
            # for param in self.model.model.encoder.parameters():
            #     param.requires_grad = False
        
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
             params=[{"params": self.model.model.encoder.parameters(), "lr": self.initial_lr},
                     {"params": self.model.model.decoder.parameters(), "lr": 10 * self.initial_lr},
                     {"params": self.model.model.segmentation_head.parameters(), "lr": 10 * self.initial_lr},
                     ], lr=self.initial_lr, weight_decay=5e-3)
        # self.optimizer = AdamW(params=self.model.model.parameters(), lr=self.initial_lr, weight_decay=5e-3)  # self.model.model
        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", factor=0.5, patience=3, threshold=0.0001, min_lr=1e-20, verbose=True)
        # iterations_per_epoch = math.ceil(len(self.train_dataset) / (self.batch_size * self.world_size))
        # total_iterations = iterations_per_epoch * self.epochs
        # self.scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        # Scaler
        self.scaler = GradScaler()

        # Checkpoint saving path
        if "baseline" in args.name.lower():
            self.model_ckpt_path = Path(f"{args.baseline_ckpt_save_loc}", f"model_{args.name}_{self.dataset_name}_{args.run_id[-7:]}.pt").absolute().__str__()
        else:
            self.model_ckpt_path = Path(f"{args.edl_ckpt_save_loc}", f"model_{args.name}_{self.dataset_name}_{args.run_id[-7:]}.pt").absolute().__str__()

        # Model Results
        self.model_results = {}

        # Metrics and results
        self.metric_names = ["loss", "mean_iou", "mean_accuracy", "overall_accuracy", "per_category_iou", "per_category_accuracy"]
        self.metric_types = ["train", "val"]
        # Train Metrics
        metrics_collection = MetricCollection({
            "iou_per_cls": JaccardIndex(task="multiclass", num_classes=self.num_classes, average="none", ignore_index=self.ignore_index),
            "acc_per_cls": Accuracy(task="multiclass", num_classes=self.num_classes, average="none", ignore_index=self.ignore_index),
            "IoU_weighted": JaccardIndex(task="multiclass", num_classes=self.num_classes, average='weighted', ignore_index=self.ignore_index),
            "acc_overall_weighted": Accuracy(task="multiclass", num_classes=self.num_classes, average='weighted', ignore_index=self.ignore_index)
        })
        self.train_metrics = metrics_collection.clone(prefix="train_").to(self.device)
        self.val_metrics = metrics_collection.clone(prefix="val_").to(self.device)
        # EDL Loss - Criterion
        self.criterion = EDLLoss(num_classes=self.num_classes, use_uncertainty=self.use_uncertainty_in_edl, use_ohem=self.use_ohem, ignore_index=self.ignore_index).to(self.device)

        # Resume check
        self.resume_training = args.resume_training

        self._summary_writer = None
        # Get TensorBoard SummaryWriter
        if self.resume_training:
            # Loading the model from checkpoint if resumed, before wrapping into DDP..
            self.log_dir = args.resume_log_dir  # Initialize summary writer here when resuming, we need purge_step taken from checkpoint
            print(f"[INFO]: Trying to resume Training from checkpoint: {self.model_ckpt_path} and log_dir: {self.log_dir}")
            loaded_dict = resume_from_checkpoint(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler, mdl_save_path=self.model_ckpt_path)
            self.model = loaded_dict["model"]
            self.optimizer = loaded_dict["optimizer"]
            self.scheduler = loaded_dict["scheduler"]
            self.scaler = loaded_dict["scaler"]
            self.best_loss = loaded_dict["best_loss"]
            self.best_mIoU = loaded_dict["best_mIoU"]
            self.start_epoch = loaded_dict["start_epoch"]
            self.model_results = loaded_dict["model_results"]  
        else:
            self.log_dir = f"runs/{args.run_id}_{args.name}"
            os.makedirs(self.log_dir, exist_ok=True)

        # we initialize summary writer only in the main process
        if self.is_main_process():
            self._summary_writer = SummaryWriter(log_dir=self.log_dir, purge_step=self.start_epoch if self.resume_training else None)  # # purge_step: Give the epoch num from which you wanna clear and restart from there. Also, Give the same log dir of the run that wanna resume
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
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "best_mIoU": self.best_mIoU,
            "epoch": self.start_epoch,
            "model_results": self.model_results
        }
        if os.path.exists(self.model_ckpt_path):
            print(f"[INFO]: Old Checkpoint File {self.model_ckpt_path} exists.. Removing it...")
            os.remove(self.model_ckpt_path)
        torch.save(obj=mld_ckpt, f=self.model_ckpt_path)
        print(f"[INFO]:Epoch {self.start_epoch} | Training checkpoint saved at {self.model_ckpt_path}.")

    def log_category_chart(self, metric_values, title, epoch):
        # Ensure metric_values is a list of tensors to handle both single last epoch and all epochs metrics
        if not isinstance(metric_values, torch.Tensor):
            raise ValueError(f"{title} metric_values is not a Tensor")
        # Convert NaNs to zeros
        metric_values[torch.isnan(metric_values)] = 0.0
        # Log as histogram (bar plot) using add_histogram
        self._summary_writer.add_histogram(f'{title}', metric_values, epoch)

    def add_metric(self, metric_type, metric_name, value):
        key = f"{metric_type}_{metric_name}"
        self.model_results.setdefault(key, []).append(value)

    def train_step(self, train_dataloader, epoch):
        self.train_metrics.reset()
        running_train_loss = 0.0
        # Training Loop
        self.model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            self.optimizer.zero_grad()
            images = batch["image"].to(self.device)                                                    # (bs, 3, h, w)
            gt_masks = batch["mask"].to(self.device)                                                   # (bs, 1, h, w)
            # Forward Pass
            # with torch.autograd.detect_anomaly():
            with autocast():
                evidences = self.model(images)                                                         # (bs, num_cls, h, w) - softplus logits
                # Evidential loss expects the target to be one-hot encoded
                labels = gt_masks.squeeze(dim=1)                                                       # (bs, h, w)
                annealing_coef = min(1.0, epoch / self.epochs)
                loss, beliefs, uncertainty, probs = self.criterion(evidences, labels, annealing_coef)  # tensor([loss]), (bs, num_cls, h, w), (bs, 1, h, w), (bs, num_cls, h, w)
            # Get the preds from probs
            preds = torch.argmax(probs, dim=1)                                                         # (bs, h, w)
            if self.is_main_process():
                print(f"\nTrain Step: {step}, Train Loss: {loss.item():.3f}")
            # Metrics calculation per batch
            running_train_loss += loss.item()
            self.train_metrics.update(preds.long(), gt_masks.squeeze(dim=1).long())
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            del batch

        # Epoch Metrics
        train_epoch_loss = running_train_loss / len(train_dataloader)
        if self.is_distributed_mode():
            train_epoch_loss = torch.tensor([train_epoch_loss], device=dist.get_rank())
            dist.all_reduce(train_epoch_loss, op=dist.ReduceOp.AVG)  # SUM(all gpus loss) / worldsize

        batch_metrics_dict = self.train_metrics.compute()
        mean_acc_epoch = torch.nanmean(batch_metrics_dict["train_acc_per_cls"])
        train_metrics_dict = {
            "loss": train_epoch_loss.item(),
            "mean_iou": batch_metrics_dict["train_IoU_weighted"].item(),
            "mean_accuracy": mean_acc_epoch.item(),
            "overall_accuracy": batch_metrics_dict["train_acc_overall_weighted"].item(),
            "per_category_iou": batch_metrics_dict["train_iou_per_cls"],
            "per_category_accuracy": batch_metrics_dict["train_acc_per_cls"],
        }
        # Logging
        metrics_dict = {k: v for k, v in train_metrics_dict.items() if k not in ["per_category_accuracy", "per_category_iou"]}
        if self.log and self.is_main_process():
            for key, value in metrics_dict.items():
                self._summary_writer.add_scalar(tag=f"Train/{key}", scalar_value=value, global_step=epoch)
        return train_metrics_dict

    def val_step(self, val_dataloader, epoch):
        self.val_metrics.reset()
        running_val_loss = 0.0
        selected_data = []
        # Eval Loop
        self.model.eval()
        for step, batch in enumerate(tqdm(val_dataloader)):
            images = batch["image"].to(self.device)
            gt_masks = batch["mask"].to(self.device)
            # Eval forward pass
            with torch.no_grad():
                evidences = self.model(images)                              # bs, num_cls, h, w
                # Evidential loss expects the target to be one-hot encoded
                labels = gt_masks.squeeze(dim=1)                            # bs, h, w
                # Loss
                annealing_coef = min(1.0, epoch / self.epochs)
                val_loss, val_beliefs, val_uncertainty, val_probs = self.criterion(evidences, labels, annealing_coef)
            # Get the preds from probs
            val_preds = torch.argmax(val_probs, dim=1)
            if self.is_main_process():
                print(f"\nVal Step: {step}, Val Loss: {val_loss.item():.3f}")
            # Metrics calculation per batch
            running_val_loss += val_loss.item()
            self.val_metrics.update(val_preds.long(), gt_masks.squeeze(dim=1).long())

            if len(selected_data) <= 1:
                idx = 0  # get the first idx of batches per step
                selected_data.append((images[idx].squeeze(), gt_masks[idx].squeeze(), val_preds[idx].squeeze()))
            del batch

        # Epoch Metrics
        val_epoch_loss = running_val_loss / len(val_dataloader)
        if self.is_distributed_mode():
            val_epoch_loss = torch.tensor([val_epoch_loss], device=dist.get_rank())
            dist.all_reduce(val_epoch_loss, op=dist.ReduceOp.AVG)  # SUM(all gpus loss) / worldsize

        batch_metrics_dict = self.val_metrics.compute()
        mean_acc_epoch = torch.nanmean(batch_metrics_dict["val_acc_per_cls"])
        val_metrics_epoch = {
            "loss": val_epoch_loss.item(),
            "mean_iou": batch_metrics_dict["val_IoU_weighted"].item(),
            "mean_accuracy": mean_acc_epoch.item(),
            "overall_accuracy": batch_metrics_dict["val_acc_overall_weighted"].item(),
            "per_category_iou": batch_metrics_dict["val_iou_per_cls"],
            "per_category_accuracy": batch_metrics_dict["val_acc_per_cls"],
        }
        # Log all metrics except per-category metrics
        metrics_dict = {k: v for k, v in val_metrics_epoch.items() if k not in ["per_category_accuracy", "per_category_iou"]}
        # Logging
        if self.log and self.is_main_process():
            for key, value in metrics_dict.items():
                self._summary_writer.add_scalar(tag=f"Val/{key}", scalar_value=value, global_step=epoch)
            # Logging of images and predictions at the end of validation epoch
            print("[INFO]: Logging Validation Images")
            if selected_data:
                plot_validation_predictions(selected_data, self._summary_writer, epoch, self.dataset_name, clip=True)
        return val_metrics_epoch

    def train(self):
        print("[INFO]: Started Training the model...")
        print(f"[INFO]: Datasets length: {len(self.train_dataset)}, {len(self.val_dataset)}")

        # DataLoaders
        train_dataloader = self._prepare_dataloader(self.train_dataset, batch_size=self.batch_size, is_distributed=self.is_distributed_mode(), is_train=self.is_train)
        val_dataloader = self._prepare_dataloader(self.val_dataset, batch_size=self.batch_size, is_distributed=self.is_distributed_mode(), is_train=self.is_train)

        # Define model_results keys dynamically, if not resumed
        if not self.resume_training:
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
            train_metrics_epoch = self.train_step(train_dataloader, self.start_epoch)
            val_metrics_epoch = self.val_step(val_dataloader, self.start_epoch)
            val_loss = val_metrics_epoch["loss"]
            self.scheduler.step(val_loss)
            # self.scheduler.step()

            if self.is_main_process():
                print(f"\nEpoch : {self.start_epoch} | Train Loss: {train_metrics_epoch['loss']} | Train mIoU: {train_metrics_epoch['mean_iou']} |  Train meanAcc: {train_metrics_epoch['mean_accuracy']} | Val Loss: {val_loss} | Val mIoU: {val_metrics_epoch['mean_iou']} | Val meanAcc: {val_metrics_epoch['mean_accuracy']} \n")

            # Saving per epoch results to model_results dict
            for metric_type in self.metric_types:
                metrics_dict = train_metrics_epoch if metric_type == "train" else val_metrics_epoch
                for metric_name in self.metric_names:
                    if metric_name in metrics_dict:
                        metric_value = metrics_dict[metric_name]
                        self.add_metric(metric_type, metric_name, metric_value)
            
            if self.is_distributed_mode():
                # Synchronize processes after each epoch
                dist.barrier()
            # Saving checkpoint of best model based on mIoU
            if val_metrics_epoch["mean_iou"] > self.best_mIoU and self.is_main_process():
                self.best_mIoU = val_metrics_epoch["mean_iou"]
                best_metrics = {"best_train_loss": train_metrics_epoch["loss"], "best_train_accuracy": train_metrics_epoch["overall_accuracy"], "best_train_mIoU": train_metrics_epoch["mean_iou"],
                                "best_val_loss": val_metrics_epoch["loss"], "best_val_accuracy": val_metrics_epoch["overall_accuracy"], "best_val_mIoU": val_metrics_epoch["mean_iou"]}
                self._save_checkpoint()
            # # Saving checkpoint of best model based on validation loss
            # if val_loss < self.best_loss and self.is_main_process():
            #     self.best_loss = val_loss
            #     best_metrics = {"best_train_loss": train_metrics_epoch["loss"], "best_train_accuracy": train_metrics_epoch["overall_accuracy"], "best_train_mIoU": train_metrics_epoch["mean_iou"],
            #                     "best_val_loss": val_metrics_epoch["loss"], "best_val_accuracy": val_metrics_epoch["overall_accuracy"], "best_val_mIoU": val_metrics_epoch["mean_iou"]}
            #     self._save_checkpoint()
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
        moved_dict = {}
        if is_final_epoch and self.is_main_process():
            print(f"[INFO]: Final Epoch: {is_final_epoch}. Logging 'per_category_iou' and 'per_category_accuracy' Bar Charts...")
            # Move the model results to cpu
            moved_dict = {key: move_to_cpu(value) for key, value in self.model_results.items()}
            if moved_dict.get("train_per_category_accuracy") is not None and self.log:
                try:
                    self.log_category_chart(moved_dict["train_per_category_accuracy"][-1], "Per Category Accuracy(Train)", self.start_epoch)
                except Exception as e:
                    print(f"[WARN]: Failed to Log Bar Charts of Train Per Category Accuracy: {e}")
            if moved_dict.get("train_per_category_iou") is not None and self.log:
                try:
                    self.log_category_chart(moved_dict["train_per_category_iou"][-1], "Per Category IoU(Train)", self.start_epoch)
                except Exception as e:
                    print(f"[WARN]: Failed to Log Bar Charts of Train Per Category IoU: {e}")
            if moved_dict.get("val_per_category_accuracy") is not None and self.log:
                try:
                    self.log_category_chart(moved_dict["val_per_category_accuracy"][-1], "Per Category Accuracy(Val)", self.start_epoch)
                except Exception as e:
                    print(f"[WARN]: Failed to Log Bar Charts of Val Per Category Accuracy: {e}")
            if moved_dict.get("val_per_category_iou") is not None and self.log:
                try:
                    self.log_category_chart(moved_dict["val_per_category_iou"][-1], "Per Category IoU(Val)", self.start_epoch)
                except Exception as e:
                    print(f"[WARN]: Failed to Log Bar Charts of Val Per Category IoU: {e}")
            print(f"[INFO]: Model Results: {moved_dict}")
            self._summary_writer.add_hparams(self.hparams, best_metrics)
            self._summary_writer.close()
        if self.is_distributed_mode():
            dist.barrier()
        return moved_dict
