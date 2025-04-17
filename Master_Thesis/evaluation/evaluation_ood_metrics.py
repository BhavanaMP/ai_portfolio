from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_auroc, binary_precision_recall_curve, binary_roc
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import auc
from torchmetrics.utilities.data import dim_zero_cat

import numpy as np
import sklearn.metrics as sk
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import plotly.graph_objects as go


class OODMetrics(Metric):
    """
    Class to calculate OOD Metrics - FPR@recalllevel, AUROC, AUPR
    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    conf: List[Tensor]
    targets: List[Tensor]

    def __init__(self, recall_level: float, pos_label: int, ignore_index: int=None, **kwargs) -> None:
        """The False Positive Rate at x% Recall or TPR metric.

        Args:
            recall_level (float): The recall level at which to compute the FPR. Usually 0.95 or 0.99.
            pos_label (int): The positive label.
            ignore_index (int, optional): Index to ignore in calculations. Defaults to None.
            kwargs: Additional arguments to pass to the metric class.

        Reference:
            Ref Link:
            - https://github.com/hendrycks/anomaly-seg
            - https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/1c906132748b5ea7fe2e1436de163397ebf4aa01/torch_uncertainty/metrics/classification/fpr95.py
        """
        super().__init__(**kwargs)

        if recall_level < 0 or recall_level > 1:
            raise ValueError(f"Recall level must be between 0 and 1. Got {recall_level}.")
        self.recall_level = recall_level
        self.pos_label = pos_label
        self.ignore_index = ignore_index
        self.add_state("conf", [], dist_reduce_fx="cat")
        self.add_state("targets", [], dist_reduce_fx="cat")

        rank_zero_warn(f"Metric `FPR{int(recall_level*100)}` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.")

    def fpr_at_tpr(self, scores, labels, recall_level=0.95):
        """
        Calculate the False Positive Rate at a certain True Positive Rate
        """
        # results will be sorted in reverse order
        fpr, tpr, thresholds = binary_roc(scores, labels)  # thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        idx = torch.searchsorted(tpr, recall_level)
        if idx == fpr.shape[0]:
            return fpr[idx - 1], thresholds[idx - 1], fpr, tpr, thresholds
        idx = np.searchsorted(tpr, recall_level, side='right')
        return fpr[idx], thresholds[idx], fpr, tpr, thresholds

    def update(self, conf: Tensor, target: Tensor) -> None:
        """Update the metric state.

        Args:
            conf (Tensor): The confidence scores.
            target (Tensor): The target labels.
        """
        self.conf.append(conf.contiguous().view(-1))
        self.targets.append(target.contiguous().view(-1))

    def compute(self) -> Tensor:
        """Compute the actual False Positive Rate at x% Recall(TPR), AUROC, AUPR

        Returns:
            Tensor: The value of the FPRx.
        """
        print("[INFO]: Computing the OOD Metrics")
        roc_curve_data, aupr_in_curve_data, aupr_out_curve_data = [], [], []
        conf = dim_zero_cat(self.conf).cpu() # .numpy()
        targets = dim_zero_cat(self.targets).cpu() # .numpy()
        print("\nOOD Metrics")
        print(f"conf.shape: {conf.shape}")
        print(f"targets.shape: {targets.shape}")

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            conf = conf * mask
            targets = targets * mask

        scores, idxs = torch.sort(conf, stable=True)
        labels = targets[idxs]
        
        print(f"scores.shape: {scores.device}, labels.device: {labels.device}")

        auroc = binary_auroc(scores, labels)
        
        # Compute AUPR for OOD as positive class. Measure the model's performance in identifying OOD samples. Represents the AUPR for identifying OOD samples (class 1).
        precision_out, recall_out, pr_thresholds_out = binary_precision_recall_curve(scores, labels)
        aupr_out = auc(recall_out, precision_out)
        aupr_out_curve_data.append(precision_out.tolist())
        aupr_out_curve_data.append(recall_out.tolist())
        aupr_out_curve_data.append(pr_thresholds_out.tolist())

        # Compute AUPR for ID as positive class. Measure the model's performance in identifying ID samples. Represents the AUPR for identifying ID samples (class 0).
        precision_in, recall_in, pr_thresholds_in = binary_precision_recall_curve(-scores, 1 - labels)  # ~scores
        aupr_in = auc(recall_in, precision_in)
        aupr_in_curve_data.append(precision_in.tolist())
        aupr_in_curve_data.append(recall_in.tolist())
        aupr_in_curve_data.append(pr_thresholds_in.tolist())

        fpr_at_recall_level, threshold_at_recall_level, fpr, tpr, thresholds_roc = self.fpr_at_tpr(scores, labels)  # (recall / TPR / Sensitivtiy) vs (FPR / 1-specificity) for different thresholds of classification scores.
        roc_curve_data.append(fpr.tolist())
        roc_curve_data.append(tpr.tolist())
        roc_curve_data.append(thresholds_roc.tolist())
        
        print(f'AUROC score: {auroc}')
        print(f'AUPR_in score: {aupr_in}, AUPR_out score: {aupr_out}')
        print(f'FPR@TPR95: {fpr_at_recall_level}')

        return fpr_at_recall_level.item(), threshold_at_recall_level.item(), aupr_in.item(), aupr_out.item(), auroc.item(), roc_curve_data, aupr_in_curve_data, aupr_out_curve_data

    def reset(self) -> None:
        """Reset the metric state."""
        self.conf = []
        self.targets = []
