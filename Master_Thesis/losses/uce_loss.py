import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics import Metric


class CELoss(nn.Module):
    """
    OhemCELoss - Online Hard Example Mining Cross Entropy Loss
    Ref: https://docs.deci.ai/super-gradients/latest/docstring/training/losses.html#latest.src.super_gradients.training.losses.ohem_ce_loss.OhemCELoss
    """

    def __init__(self, id2label, weights: torch.Tensor = None, threshold: float = 0.8, # 0.7
                 mining_percent: float = 0.7, min_kept: int = 350000, ignore_index: int = 255,
                 use_uncertainty_weighting: bool = False, use_ohem: bool = False, alpha: int = 10,
                 device: torch.device = torch.device('cpu')):
        """
        Args:
            threshold: Sample below probability threshold, is considered hard. 
                       If loss of pixel is > threshold, its considered hard
                Eg: -torch.log(torch.Tensor([0.8])) = tensor([0.2231])
                    -torch.log(torch.Tensor([0.7])) = tensor([0.3567]) 
                    -torch.log(torch.Tensor([0.4])) = tensor([0.9163])
                    -torch.log(torch.Tensor([0.3])) = tensor([1.2040])
                    -torch.log(torch.Tensor([0.2])) = tensor([1.6094])

            ignore_index: label index to be ignored in loss calculation
            mining_percent: Total percentage of pixels to be retained in every backward pass
        """
        super().__init__()

        if mining_percent < 0 or mining_percent > 1:
            raise ValueError(f"mining percent should be between (0, 1] but given {mining_percent}")

        self.device = device
        self.thresh = -torch.log(torch.tensor(threshold, dtype=torch.float, device=self.device))

        self.mining_percent = mining_percent
        self.min_kept = max(1, min_kept)
        self.ignore_index = ignore_index

        # If weights are not provided, initialize with equal weights for all classes
        num_classes = len(id2label)
        if isinstance(weights, list):  # Weights provided as a list
            if len(weights) != num_classes:
                raise ValueError(f"Weights are given for {len(weights)} classes, but {num_classes} are needed")
            self.class_weights = torch.FloatTensor(weights).to(self.device)
        else:
            self.class_weights = torch.ones(num_classes).to(self.device)

        self.use_ohem = use_ohem
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.ce_criterion = CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index).to(self.device)
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index, reduction="none").to(self.device)  # https://discuss.pytorch.org/t/softmax-cross-entropy-loss/125383

    def forward(self, logits, labels, predictive_uncertainty: torch.Tensor = None, use_percentage_logic: bool = True, return_class_wise_loss: bool=False):
        if self.use_uncertainty_weighting and predictive_uncertainty is not None:
            # MCD Training
            return self.compute_mcd_uncertainty_loss(logits, labels, predictive_uncertainty, use_percentage_logic, return_class_wise_loss)
        if self.use_ohem:
            # When not using MCD Training and but OHEM is set, applies for baseline training with OHEM
            return self.compute_ohem_loss(logits, labels, use_percentage_logic, return_class_wise_loss)
        else:
            # Usual Baseline with no MCD, no OHEM returns usual Cross Entropy Loss
            return self.ce_criterion(logits, labels)

    def flatten_and_filter(self, loss, labels):
        loss_flat = loss.contiguous().view(-1)
        labels_flat = labels.contiguous().view(-1)
        mask = labels_flat != self.ignore_index
        return loss_flat[mask], labels_flat[mask]

    def compute_mcd_uncertainty_loss(self, logits, labels, predictive_uncertainty, use_percentage_logic, return_class_wise_loss):
        print("[INFO]: Calculating uncertainty loss as predictive uncertainty is not none")
        weighted_loss = ((1 + predictive_uncertainty.squeeze()) ** self.alpha) * self.criterion(logits, labels)
        loss_flat, labels_flat = self.flatten_and_filter(weighted_loss, labels)
        if self.use_ohem:
            print("[INFO]: Invoking OHEM with MCD")
            loss_mean = self.apply_ohem(loss_flat, labels_flat, device=logits.device, use_percentage_logic=use_percentage_logic, size=labels.size(0))
        else:
            # Return the reduced weighted loss of MCD without OHEM
            loss_mean = loss_flat.sum() / self.class_weights[labels_flat].sum()
        if return_class_wise_loss:
            class_wise_loss = self.get_class_wise_losses(loss_flat, labels_flat, num_classes=logits.size(1), device=logits.device)
            return loss_mean, class_wise_loss
        return loss_mean

    def compute_ohem_loss(self, logits, labels, use_percentage_logic, return_class_wise_loss):
        print("[INFO]: Invoking OHEM for baseline")
        loss = self.criterion(logits, labels)
        loss_flat, labels_flat = self.flatten_and_filter(loss, labels)
        loss_mean = self.apply_ohem(loss_flat, labels_flat, device=logits.device, use_percentage_logic=use_percentage_logic, size=labels.size(0))
        if return_class_wise_loss:
            class_wise_loss = self.get_class_wise_losses(loss_flat, labels_flat, num_classes=logits.size(1), device=logits.device)
            return loss_mean, class_wise_loss
        return loss_mean

    def apply_ohem(self, loss_flat, labels_flat, device, use_percentage_logic=True, size=None):
        # Check if any pixels are available for mining, if not, return empty loss tensor
        num_pixels = loss_flat.numel()
        if num_pixels == 0:
            return torch.tensor([0.0], device=device).requires_grad_(True)
        self.thresh = self.thresh.to(device)
        # Calculate the number of pixels to be selected for mining
        if use_percentage_logic:
            print("[INFO]: Invoking percentage logic")
            num_mining = int(self.mining_percent * num_pixels)
            # in case mining_percent=1, prevent out of bound exception
            num_mining = max(1, min(num_mining, num_pixels - 1))
        else:
            print("[INFO]: Invoking min_kept logic")
            batch_kept = int(self.min_kept * size)
            num_mining = max(1, min(batch_kept, num_pixels-1))
        # Sort the loss values in descending order
        sorted_loss, indices = torch.sort(loss_flat, descending=True)
        sorted_labels = labels_flat[indices]
        # Calculate the threshold dynamically for selecting hard examples
        threshold = max(self.thresh, sorted_loss[num_mining])
        # Check if the loss values meet the threshold
        hard_examples_mask = sorted_loss >= threshold
        if hard_examples_mask.sum() < num_mining:
            # If the number of hard examples is less than num_mining, select the top num_mining examples
            selected_loss = sorted_loss[:num_mining]
            selected_labels = sorted_labels[:num_mining]
        else:
            # Select the hard examples based on the mask
            selected_loss = sorted_loss[hard_examples_mask][:num_mining]
            selected_labels = sorted_labels[hard_examples_mask][:num_mining]
        # Calculate the weighted mean loss
        loss_mean = selected_loss.sum() / self.class_weights[selected_labels].sum()
        return loss_mean

    def get_class_wise_losses(self, loss, labels, num_classes, device):
        class_wise_loss = torch.zeros(num_classes, device=device)
        for c in range(num_classes):
            class_mask = labels == c
            # Calculate class-wise loss by averaging over pixels of the current class and multiply the loss by the class weight for the current class
            class_loss = torch.mean(loss[class_mask] * self.class_weights[c]) if torch.any(class_mask) else torch.tensor(0.0, device=device)
            class_wise_loss[c] = class_loss.detach()
        return class_wise_loss


class ClassWiseLoss(Metric):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("class_losses", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("num_update_steps", default=torch.Tensor([0]), dist_reduce_fx="sum")

    def update(self, class_losses):
        # Accumulate class losses for each batch
        self.class_losses += class_losses
        # Increment the number of update steps
        self.num_update_steps += 1

    def compute(self):
        # Divide accumulated class losses by the number of update steps (batches) to get the mean
        return self.class_losses / self.num_update_steps

    def reset(self):
        # Reset accumulated losses and update step count
        self.class_losses.zero_()
        self.num_update_steps.zero_()
