import copy

import torch
from torch import nn
import torch.nn.functional as F

from modelling.model_utils import enable_dropout, get_smp_model


class MCDClassifier(nn.Module):
    """
    Monte Carlo Dropout  Deep Learning Classifier
    """
    def __init__(self, num_classes, encoder, decoder, train_mcd=False, forward_passes=0, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.model = get_smp_model(num_classes=num_classes, encoder=encoder, decoder=decoder)
        self.train_mcd = train_mcd
        self.forward_passes = forward_passes
        if isinstance(device, int):  # Check if device is an integer
            self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):  # Check if device is a string
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images, is_val=False):
        logits = self.model(images)
        if logits.shape[2:] != images.shape[2:]:
            logits = F.interpolate(logits, size=images.shape[2:], mode="nearest")
        if self.train_mcd:
            predictive_uncertainty = self.calculate_predictive_uncertainty(images, is_val=is_val)
            return logits, predictive_uncertainty
        return logits

    @torch.no_grad()
    def calculate_predictive_uncertainty(self, images, is_val=False):
        print(f"[INFO]: Performing forward pass of MC Dropout... {self.model.training}, {torch.is_grad_enabled()}")
        mc_logits = torch.empty(size=(self.forward_passes, len(images), self.num_classes, images.size(-2), images.size(-1)), device=self.device)

        for i in range(self.forward_passes):
            # Enabling dropout for the model in eval mode
            if is_val:
                print("[INFO]: Enabling dropout layers for Val Step to get predictive uncertainty")
                model = copy.deepcopy(self.model)
                model = enable_dropout(model)
                mylogits = model(images)
            else:
                mylogits = self.model(images)
            mc_logits[i] = mylogits
            del mylogits
        y_prob_samples = torch.softmax(mc_logits, dim=2)
        preds_probs = torch.mean(y_prob_samples, dim=0)
        preds_var = torch.std(torch.softmax(mc_logits, dim=2), dim=0)
        preds = torch.argmax(preds_probs, dim=1)
        predictive_uncertainty = torch.empty((images.shape[0], images.shape[2], images.shape[3]), device=self.device)
        for i in range(self.num_classes):
            predictive_uncertainty = torch.where(preds == i, preds_var[:, i, :, :], predictive_uncertainty)
        return predictive_uncertainty
