import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from segmentation_models_pytorch.losses import DiceLoss, FocalLoss


# DDP Compatibility Classes and Functions
class DistInfo:
    """
    Ref Link: https://github.com/pytorch/pytorch/issues/98286
    """
    is_parallel = False
    local_rank = 0
    world_size = 1

    def __init__(self):
        if dist.is_initialized():
            DistInfo.is_parallel = True
            DistInfo.local_rank = dist.get_rank()
            DistInfo.world_size = dist.get_world_size()

    @classmethod
    def init(cls):
        """
        Info Link: 
        - https://sentry.io/answers/difference-between-staticmethod-and-classmethod-function-decorators-in-python/#:~:text=We%20can%20decorate%20a%20function,(%22Alice%22)%20alice.
        - https://web.archive.org/web/20140307141816/http://julien.danjou.info/blog/2013/guide-python-static-class-abstract-methods
        """
        if dist.is_initialized():
            cls.is_parallel = True
            cls.local_rank = dist.get_rank()
            cls.world_size = dist.get_world_size()


def cat_all_gather(input_):
    """
    Gather all the tensors from all the processes and gather.
    """
    if not isinstance(input_, torch.Tensor):
        raise TypeError(f"input must is a torch.Tensor, but input is {type(input_)}")
    if DistInfo.world_size == 1:
        return input_
    gather_list = [torch.empty_like(input_) for _ in range(DistInfo.world_size)]
    dist.all_gather(gather_list, input_)
    gather_list[DistInfo.local_rank] = input_
    output = torch.cat(gather_list, 0).contiguous()
    return output


# Loss Functions for EDL
class TypeIIMaximumLikelihoodLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Dirichlet distribution D(p|alphas) is used as prior on the likelihood of Multi(y|p).
            Ref: https://github.com/clabrugere/evidential-deeplearning
                 https://muratsensoy.github.io/uncertainty.html
        """
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=1, keepdim=True)

        loss = torch.sum(labels * (torch.log(strength) - torch.log(alphas)), dim=1)

        return torch.mean(loss)


class CEBayesRiskLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Bayes risk is the maximum cost of making incorrect estimates, taking a cost function assigning a penalty of
        making an incorrect estimate and summing it over all possible outcomes. Here the cost function is the Cross Entropy.
        """
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        alphas = evidences + 1.0
        strengths = torch.sum(alphas, dim=1, keepdim=True)

        loss = torch.sum(labels * (torch.digamma(strengths) - torch.digamma(alphas)), dim=1)

        return torch.mean(loss)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, labels_one_hot):
        print(probs.shape, labels_one_hot.shape)
        # Calculate intersection and union
        intersection = (probs * labels_one_hot).sum()
        union = probs.sum() + labels_one_hot.sum()

        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        print(dice.shape, dice)
        return 1 - dice


class SSBayesRiskLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Same as CEBayesRiskLoss but here the cost function is the sum of squares instead."""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels, use_uncertainty, uncertainty_cap=1.0, epsilon=1e-8, penalty_coefficient=1.0):       
        num_classes = evidences.size(1)                                         # Evidences: ([bs, num_cls, h, w]), Labels: ([bs, num_cls, h, w])

        alphas = evidences + 1.0                                                # ([bs, num_cls, h, w])
        strength = torch.sum(alphas, dim=1, keepdim=True)                       # ([bs, 1, h, w])
        probabilities = alphas / strength                                       # ([bs, num_cls, h, w])

        error = (labels - probabilities) ** 2                                    # ([bs, num_cls, h, w])
        variance = probabilities * (1.0 - probabilities) / (strength + 1.0)      # ([bs, num_cls, h, w])

        beliefs = evidences / strength                                           # ([bs, num_cls, h, w])
        uncertainty = num_classes / strength                                     # ([bs, 1, h, w])
        
        loss = torch.sum(error + variance, dim=1)                                # ([bs, h, w])

        # Penalizing overconfidence
        # overconfidence_penalty = penalty_coefficient * (1 - uncertainty) ** 2
        # loss = torch.sum(error + variance + overconfidence_penalty, dim=1)
        
        # Use Uncertainty Weighting
        if use_uncertainty:
            print("[INFO]: Calculating uncertainty weighting EDL loss")
            # Squeeze uncertainty to match loss dimensions
            uncertainty = uncertainty.squeeze(dim=1)                             # (bs, 1, h, w) to (bs, h, w)
            
            # # Cap the uncertainty to avoid excessively large values
            # capped_uncertainty = torch.clamp(uncertainty, max=uncertainty_cap)
            
            # Add epsilon to avoid division by zero or excessively small values
            # loss = ((1 + capped_uncertainty) ** 2 + epsilon) * loss               # (bs, h, w)  # capped_uncertainty
            loss = ((1 + uncertainty.squeeze(dim=1)) ** 2) * loss                 # (bs, h, w), (bs, h, w)
        
        return loss, beliefs, uncertainty, probabilities  # torch.mean(loss)


class KLDivergenceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        KL divergence is a measure of how one probability distribution diverges from a second, expected probability
        distribution. It is often used to measure the difference between two probability distributions.
        Regularization: It acts as a regularization term, helping to encourage the model to produce 
        outputs (probability distributions) that align more closely with the expected distribution 
        (often a uniform distribution or a target distribution). It is typically applied in scenarios where the 
        model output (e.g., predicted probabilities) should match a target distribution or where we want to penalize 
        deviations from a prior belief distribution. Here, KLD acts as a regularization term to shrink the evidence 
        of samples towards zero that cannot be correctly classified. Also, KLD loss is annealed, meaning its
        influence on the overall loss is gradually increased or decreased during training.
        This is done to balance the learning process and prevent the model from focusing too much on regularization
        early in training.
        """
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        num_classes = evidences.size(1)                                                               # Evidences: ([bs, num_cls, h, w]), Labels: ([bs, num_cls, h, w])

        alphas = evidences + 1.0                                                                      # ([bs, num_cls, h, w])
        alphas_tilde = labels + (1.0 - labels) * alphas                                               # ([bs, num_cls, h, w])
        strength_tilde = torch.sum(alphas_tilde, dim=1, keepdim=True)                                 # ([bs, 1, h, w])

        # lgamma is the log of the gamma function   
        first_term = (
            torch.lgamma(strength_tilde) - torch.lgamma(alphas_tilde.new_tensor(num_classes, dtype=torch.float32)) - (torch.lgamma(alphas_tilde)).sum(dim=1, keepdim=True)
        )                                                                                             # ([bs, 1, h, w])
        second_term = torch.sum(
            (alphas_tilde - 1.0) * (torch.digamma(alphas_tilde) - torch.digamma(strength_tilde)), dim=1, keepdim=True
        )                                                                                             # ([bs, 1, h, w])
        loss = first_term.squeeze(dim=1) + second_term.squeeze(dim=1)                                 # ([bs, h, w])

        return loss  # torch.mean(first_term + second_term)

class EDLLoss(nn.Module):
    """
    EDL Loss
    """
    def __init__(self, num_classes, use_uncertainty: bool = False, alpha: int = 10, ignore_index: int = 255, use_ohem: bool = False, threshold: float = 0.8, mining_percent: float = 0.7, **kwargs):
        """Uses SS_BayesRiskLoss and KLDivergence Loss. Third Loss from the paper"""
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Uncertainty weihting
        self.use_uncertainty = use_uncertainty
        self.alpha = alpha
        # OHEM
        self.use_ohem = use_ohem
        self.mining_percent = mining_percent
        if mining_percent < 0 or mining_percent > 1:
            raise ValueError(f"mining percent should be between (0, 1] but given {mining_percent}")
        self.thresh = -torch.log(torch.tensor(threshold, dtype=torch.float))
        # Loss functions
        self.ssbayesrisk_loss = SSBayesRiskLoss()
        self.kld_loss = KLDivergenceLoss()
        # self.dice_loss = DiceLoss()
        # self.focal_loss = FocalLoss(mode="multiclass", gamma=2, ignore_index=self.ignore_index, reduction=None)
        # DDP Compatibility
        self.dist_info = DistInfo()

    def forward(self, evidences, labels, annealing_coef):
        assert labels.ndim == 3, f"Labels should have a shape of (bs, h, w) but found shape: {labels.shape}"

        # Note: You do not need to gather/concat labels. The loss is calculated on every GPU and gradients are gathered, averaged and then backpropagated on all GPUs by DDP.

        # Mask out the ignore_index values to handle onehot encoding. Otherwise cuda throws error cuz of 255 in the labels while one hot encoding.
        valid_mask = labels != self.ignore_index
        valid_labels = labels.clone()
        # Temporarily set ignore_index to a valid class index
        valid_labels[~valid_mask] = 0

        # Evidential loss expects the target to be one-hot encoded. One-hot encode the valid labels
        labels_one_hot = F.one_hot(valid_labels.long(), num_classes=self.num_classes)               # bs, h, w, num_cls
        labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).contiguous()                            # bs, num_cls, h, w

        # Restore the ignore_index in the one-hot encoded labels
        labels_one_hot = labels_one_hot.float()
        # Expand mask to match labels_one_hot shape
        expanded_mask = valid_mask.unsqueeze(1).expand(-1, self.num_classes, -1, -1)
        # Set ignore index positions to zero
        labels_one_hot[~expanded_mask] = 0

        # Sum of Squares Bayes Risk Loss
        mse_loss, beliefs, uncertainty, probs = self.ssbayesrisk_loss(evidences, labels_one_hot, self.use_uncertainty)

        # KL Divergence Loss
        kl_loss = self.kld_loss(evidences, labels_one_hot)
        
        # # Dice loss
        # dice_loss = self.dice_loss(probs, labels_one_hot)
        
        # # Focal Loss
        # focal_loss = self.focal_loss(probs, labels)  # Filtering happens in KL Loss directly
        # print(focal_loss.shape)

        # Combine mean losses usually use_ohem ohem or use_uncertainty is not set
        if not self.use_ohem:   # and not self.use_uncertainty
            # Filter the ignore_index
            mse_loss, _ = self.filter(mse_loss, labels)
            kl_loss, _ = self.filter(kl_loss, labels)

            # calculate the loss
            loss = torch.mean(mse_loss) + annealing_coef * torch.mean(kl_loss)
            # loss = torch.mean(mse_loss) + 1.5 * dice_loss + annealing_coef * torch.mean(kl_loss)
            # loss = torch.mean(mse_loss) + torch.mean(focal_loss) + annealing_coef * torch.mean(kl_loss)
            # loss = torch.mean(mse_loss)
            return loss, beliefs, uncertainty, probs
        else:
            # FIXME: Handle dice loss here
            loss = mse_loss + annealing_coef * kl_loss                    # (bs, h, w), (bs, h, w) = (bs, h, w)
            # Filter the ignore_index
            loss, labels = self.filter(loss, labels)

            print("[INFO]: Invoking OHEM in EDL")
            loss_flat, labels_flat = self.flatten(loss, labels)
            loss_mean = self.apply_ohem(loss_flat, labels_flat, device=evidences.device)
            return loss_mean, beliefs, uncertainty, probs

    def filter(self, loss, labels):
        mask = labels != self.ignore_index
        filtered_loss = loss * mask
        filtered_labels = labels * mask
        return filtered_loss, filtered_labels

    def flatten(self, loss, labels):
        loss_flat = loss.contiguous().view(-1)
        labels_flat = labels.contiguous().view(-1)
        return loss_flat, labels_flat

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
        sorted_loss, _ = torch.sort(loss_flat, descending=True)
        # Calculate the threshold dynamically for selecting hard examples
        threshold = max(self.thresh, sorted_loss[num_mining])
        # Check if the loss values meet the threshold
        hard_examples_mask = sorted_loss >= threshold
        if hard_examples_mask.sum() < num_mining:
            # If the number of hard examples is less than num_mining, select the top num_mining examples
            selected_loss = sorted_loss[:num_mining]
        else:
            # Select the hard examples based on the mask
            selected_loss = sorted_loss[hard_examples_mask][:num_mining]
        # Calculate the weighted mean loss
        loss_mean = selected_loss.mean()
        return loss_mean
