from typing import Literal, Optional, Any, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.classification import MulticlassCalibrationError

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import plotly.graph_objects as go
import plotly.io as pio   
# pio.kaleido.scope.mathjax = None
from plotly.subplots import make_subplots



class ClassWiseUncertainty(Metric):
    def __init__(self, num_classes, ignore_index=None, **kwargs):
        """
        This class can be used to get class wise entropy / class wise uncertainty of origianl / mc logits.

        Just pass in the required value in place of predictive_uncertainty while calling the update function.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state("class_wise_total_uncertainty", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("class_wise_sample_counts", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds, target, predictive_uncertainty):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            preds = preds * mask
        for i in range(self.num_classes):
            # Mask to select predictions for class i
            mask = preds == i
            class_uncertainty = predictive_uncertainty[mask]
            if class_uncertainty.numel() > 0:
                # Sum of uncertainties for class i
                total_uncertainty = class_uncertainty.sum()
                self.class_wise_total_uncertainty[i] += total_uncertainty.item()
                self.class_wise_sample_counts[i] += mask.sum().item()

    def compute(self):
        # Calculate mean uncertainty per class
        class_wise_mean_uncertainty = torch.zeros_like(self.class_wise_total_uncertainty)
        # Avoid division by zero
        valid_class_counts = torch.clamp(self.class_wise_sample_counts, min=1)
        class_wise_mean_uncertainty = self.class_wise_total_uncertainty / valid_class_counts.float()

        return class_wise_mean_uncertainty

    def reset(self):
        self.class_wise_total_uncertainty.zero_()
        self.class_wise_sample_counts.zero_()


class ClassWiseFrequency(Metric):
    def __init__(self, num_classes, ignore_index=None, **kwargs):
        """
        This class can be used to get class wise pixel frequency.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state("class_wise_counts", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            preds = preds * mask
        for i in range(self.num_classes):
            # Mask to select predictions for class i
            mask = preds == i
            class_count = mask.sum().item()
            self.class_wise_counts[i] += class_count

    def compute(self):
        total_counts = self.class_wise_counts.sum().item()
        class_wise_frequencies = self.class_wise_counts / total_counts
        return class_wise_frequencies

    def reset(self):
        self.class_wise_counts.zero_()


class NegativeLogLikelihood(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, reduction: Literal["mean", "sum", "none", None] = "mean", ignore_index: Optional[int] = None, **kwargs: Any,) -> None:
        """The Negative Log Likelihood Metric.
        Ref: https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/1c906132748b5ea7fe2e1436de163397ebf4aa01/torch_uncertainty/metrics/classification/categorical_nll.py
        Args:
            reduction (str, optional): Determines how to reduce over the batch dimension
                - 'mean' [default]: Averages score across pixels
                - 'sum': Sum score across pixels
                - 'none' or None: Returns score per pixel
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient.
        Params:
            probs: (bs, num_classes, h, w) - softmax probs
            target: (bs, h, w)
        """
        super().__init__(**kwargs)
        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )
        self.reduction = reduction
        self.ignore_index = ignore_index

        if self.reduction in ["mean", "sum"]:
            self.add_state("values", default=torch.tensor(0.0), dist_reduce_fx="sum",)
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: Tensor, target: Tensor) -> None:
        """Compute the negative log likelihood loss."""
        bs, h, w = target.shape
        if self.reduction is None or self.reduction == "none":
            self.values.append(F.nll_loss(torch.log(probs), target, reduction="none", ignore_index=self.ignore_index))
        else:
            self.values += F.nll_loss(torch.log(probs), target, reduction="sum", ignore_index=self.ignore_index)
            self.total += (target != self.ignore_index).sum()  # bs*h*w

    def compute(self) -> Tensor:
        if self.reduction in ["sum", "mean"]:
            if self.reduction == "sum":
                return self.values
            elif self.reduction == "mean":
                return self.values / self.total
        else:
            # reduction is None or "none"
            values = dim_zero_cat(self.values)
            return values

    def reset(self):
        # Reinitialize the tensor with zeros
        self.total.zero_()
        if self.reduction in ["mean", "sum"]:
            self.values.zero_()
        else:
            self.values = []


def compute_aurrc_selection_thresholds(preds, selection_scores, num_bins: int=10):

    print(f"[INFO]: Computing Selection Threshold of AURRC, selection_scores: {selection_scores.shape}, preds: {preds.shape}")
    
    assert preds.shape == selection_scores.shape, f"Shapes Mismatch preds: {preds.shape}, selection_scores: {selection_scores.shape}"
    
    total_samples = len(preds)
    
    # Sort selection_scores tensor in descending order
    order = np.argsort(selection_scores)[::-1]

    # To handle cases where the number of bins exceeds the number of samples
    if num_bins > total_samples:
        raise ValueError(f"Number of bins {num_bins} exceeds the number of samples {total_samples}")
    
    bin_indices = []
    selection_thresholds = []

    for bin_id in range(num_bins):
        samples_in_bin = total_samples // num_bins
        selection_threshold = selection_scores[order[samples_in_bin * (bin_id + 1) - 1]]
        selection_thresholds.append(selection_threshold)
        ids = selection_scores >= selection_threshold
        # print(type(ids))  # ndarray
        # print(ids.shape)  # 932469500
        bin_indices.append([ids])
        
    return selection_thresholds, bin_indices


def compute_aurrc_risks_rejections(preds, target, bin_indices, risk_func):
    print(f"[INFO]: Computing Risks and Rejection of AURRC, preds: {preds.shape}, target: {target.shape}")
    
    risks, rejection_rates = [], []
    
    for ids in bin_indices:
        print(ids[0].shape)
        selected_preds = preds[ids[0]]
        selected_target = target[ids[0]]
        if np.sum(ids[0]) > 0:  # list(ndarray)
            risk_value = 1.0 - risk_func(selected_target, selected_preds)
        else:
            risk_value = 0.0

        risks.append(risk_value)
        rejection_rates.append(1.0 - 1.0 * np.sum(ids[0]) / len(target))
    
    aurrrc = np.nanmean(risks)

    return aurrrc, rejection_rates, risks


# def area_under_risk_rejection_rate_curve(probs=None, target=None, preds=None, selection_scores=None, risk_func=accuracy_score, num_bins: int=10, ignore_index: int=None, return_counts: bool=True):
#     """ Computes risk vs rejection rate curve and the area under this curve. Similar to risk-coverage curves [3]_ where
#     coverage instead of rejection rate is used.

#     Code Ref Link: https://github.com/IBM/UQ360/blob/main/uq360/metrics/classification_metrics.py

#     References:
#         .. [3] Franc, Vojtech, and Daniel Prusa. "On discriminative learning of prediction uncertainty."
#          In International Conference on Machine Learning, pp. 1963-1971. 2019.

#     Args:
#         target: ground truth labels. array-like of shape - (bs, height, width)
#         probs: Probability scores from the base model. array-like of shape - (bs, num_classes, height, width)
#         preds: predicted labels.array-like of shape - (bs, height, width)
#         selection_scores: scores corresponding to certainty in the predicted labels.
#         risk_func: risk function under consideration.
#         num_bins: number of bins.
#         return_counts: set to True to return counts also.
#         ignore_index: Label to ignore in the calculations.

#     Returns:
#         float or tuple:
#             - aurrrc (float): area under risk rejection rate curve.
#             - rejection_rates (list): rejection rates for each bin (returned only if return_counts is True).
#             - selection_thresholds (list): selection threshold for each bin (returned only if return_counts is True).
#             - risks (list): risk in each bin (returned only if return_counts is True).

#     """
    
#     if selection_scores is None:
#         if preds is None:
#             # Get the selection scores and indices of maximum probability
#             selection_scores, preds = torch.max(probs, dim=1)
#         else:
#             selection_scores, _ = torch.max(probs, dim=1)
#             print(probs.shape, target.shape)
    
#     # Concatenate all batches
#     preds = torch.cat(preds, dim=0)
#     selection_scores = torch.cat(selection_scores, dim=0)

#     # print(f"selection_scores: {selection_scores.shape}, preds: {preds.shape}, target: {target.shape}")
    
#     # bs, num_classes, height, width = probs.shape
#     bs, height, width = preds.shape
#     total_samples = bs * height * width
    
#     # Flatten and convert to ndArray
#     selection_scores = selection_scores.contiguous().view(-1).cpu().numpy()
#     preds = preds.contiguous().view(-1).cpu().numpy()
    
#     print(f"selection_scores: {selection_scores.shape}, preds: {preds.shape}")
    
#     target = torch.cat(target, dim=0)
#     target = target.contiguous().view(-1).cpu().numpy()
    
#     # Filter out the ignore index
#     valid_indices = target != ignore_index
#     selection_scores = selection_scores * valid_indices
#     preds = preds * valid_indices
#     target = target * valid_indices

#     # Sort selection_scores tensor in descending order
#     order = np.argsort(selection_scores)[::-1]

#     assert preds.shape == target.shape == selection_scores.shape, f"Shapes Mismatch preds: {preds.shape}, target: {target.shape}, selection_scores: {selection_scores.shape}"

#     rejection_rates, selection_thresholds, risks = [], [], []

#     # To handle cases where the number of bins exceeds the number of samples
#     if num_bins > total_samples:
#         raise ValueError(f"Number of bins {num_bins} exceeds the number of samples {total_samples}")

#     for bin_id in range(num_bins):
#         samples_in_bin = total_samples // num_bins
#         selection_threshold = selection_scores[order[samples_in_bin * (bin_id + 1) - 1]]
#         selection_thresholds.append(selection_threshold)
#         ids = selection_scores >= selection_threshold
#         if sum(ids) > 0:
#             risk_value = 1.0 - risk_func(target[ids], preds[ids])
#         else:
#             risk_value = 0.0

#         risks.append(risk_value)
#         rejection_rates.append(1.0 - 1.0 * sum(ids) / len(target))

#     aurrrc = np.nanmean(risks)

#     if not return_counts:
#         return aurrrc
#     else:
#         return aurrrc, rejection_rates, selection_thresholds, risks

# ECE..add reliability plot to original ECE
def _custom_binning_bucketize(confidences: Tensor, accuracies: Tensor, bin_boundaries: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute calibration bins using ``torch.bucketize``. Use for ``pytorch >=1.6``.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities

    """
    accuracies = accuracies.to(dtype=confidences.dtype)
    acc_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
    conf_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
    count_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)

    indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1

    count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

    conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
    conf_bin = torch.nan_to_num(conf_bin / count_bin)

    acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
    acc_bin = torch.nan_to_num(acc_bin / count_bin)

    return acc_bin, conf_bin, count_bin

def _custom_ce_compute(confidences: Tensor, accuracies: Tensor, bin_boundaries: Union[Tensor, int], norm: str = "l1", debias: bool = False,) -> Tensor:
    """Compute the calibration error given the provided bin boundaries and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
        norm: Norm function to use when computing calibration error. Defaults to "l1".
        debias: Apply debiasing to L2 norm computation as in
            `Verified Uncertainty Calibration`_. Defaults to False.

    Raises:
        ValueError: If an unsupported norm function is provided.

    Returns:
        Tensor: Calibration error scalar.

    """
    if isinstance(bin_boundaries, int):
        bin_boundaries = torch.linspace(0, 1, bin_boundaries + 1, dtype=confidences.dtype, device=confidences.device)

    if norm not in {"l1", "l2", "max"}:
        raise ValueError(f"Argument `norm` is expected to be one of 'l1', 'l2', 'max' but got {norm}")

    with torch.no_grad():
        acc_bin, conf_bin, count_bin = _custom_binning_bucketize(confidences, accuracies, bin_boundaries)
        prop_bin = count_bin / count_bin.sum()

        avg_acc = torch.sum(acc_bin * count_bin) / torch.sum(count_bin)
        avg_conf = torch.sum(conf_bin * count_bin) / torch.sum(count_bin)

    res = {
        "accuracies": acc_bin, "confidences": conf_bin,
        "counts": count_bin, "avg_accuracy": avg_acc,
        "avg_confidence": avg_conf, "bins": bin_boundaries
    }
    if norm == "l1":
        ece = torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        res["expected_calibration_error"] = ece
    if norm == "max":
        mce = torch.max(torch.abs(acc_bin - conf_bin))
        res["max_calibration_error"] = mce
    if norm == "l2":
        ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
        # NOTE: debiasing is disabled in the wrapper functions. This implementation differs from that in sklearn.
        if debias:
            # the order here (acc_bin - 1 ) vs (1 - acc_bin) is flipped from the equation in Verified Uncertainty Prediction (Kumar et al 2019)/
            debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (prop_bin * accuracies.size()[0] - 1)
            ce += torch.sum(torch.nan_to_num(debias_bins))  # replace nans with zeros if nothing appeared in a bin
        l2_calibration_error = torch.sqrt(ce) if ce > 0 else torch.tensor(0)
        res["l2_calibration_error"] = l2_calibration_error
    return res

class CustomMultiClassCalibrationError(MulticlassCalibrationError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self) -> Tensor:
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        bin_data = _custom_ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)
        return bin_data

    @staticmethod
    def ensure_numpy(a):
        if not isinstance(a, np.ndarray):
            a = a.cpu().numpy()
        return a

    def _confidence_histogram_subplot(self, fig, bin_data, draw_averages, title="Examples per bin", xlabel="Confidence", ylabel="Count", show_axis_titles=True):
        """Draws a confidence histogram into a subplot."""
        counts = bin_data["counts"]
        bins = bin_data["bins"]
        bin_size = 1.0 / len(counts)
        positions = bins[:-1] + bin_size / 2.0
        # Add the bar trace to the figure
        bar_trace = go.Bar(x=positions, y=counts, width=bin_size * 0.9, name="Examples per bin", marker=dict(color='blue'))

        # Determine the scale factor for large numbers
        max_count = np.max(np.abs(counts))  # Taking absolute since counts are negative for inverted histogram
        if max_count >= 1e9:
            scale_factor = 1e9  # Billions
            tick_suffix = 'B'
        elif max_count >= 1e6:
            scale_factor = 1e6  # Millions
            tick_suffix = 'M'
        else:
            scale_factor = 1  # No scaling, keep original
            tick_suffix = ''
        
        # Update Y axis scale
        range_vl = [0, np.max(np.abs(counts))]  # Get the range of negative counts
        # Safeguard against invalid tick distances
        distance_tick = (range_vl[1] - range_vl[0]) / 5  # Set 5 ticks across the range
        # distance_tick = max(range_vl[1] / 5, 1)  # Ensure at least one tick

        # Create tick values as negative (for inversion) but display them as positive
        tick_vals = np.arange(range_vl[0], range_vl[1] + distance_tick, distance_tick)
        tick_text = [f"{int(val / scale_factor)}{tick_suffix}" for val in tick_vals]  # Convert negative values to positive for display

        # Update the layout
        fig.update_layout(title=title,
                          xaxis2=dict(title=xlabel if show_axis_titles else None,
                                      showline=True, linewidth=1, linecolor='black', mirror=True,
                                      showgrid=True, gridcolor="lightgray",
                                      title_font=dict(size=18),  # Increase xlabel font size
                                       tickfont=dict(size=14)  # Increase tick font size for x-axis
                                    ),  # xlabel
                          yaxis2=dict(title="" if show_axis_titles else None, # ylabel
                                      tickmode="array", tickvals=tick_vals, ticktext=tick_text,
                                      showline=True, linewidth=1, linecolor='black', mirror=True,
                                      showgrid=True, gridcolor="lightgray",
                                      title_font=dict(size=18),  # Increase ylabel font size
                                      tickfont=dict(size=14)  # Increase tick font size for y-axis
                                      ),
                          legend=dict(title="", x=0, y=1, traceorder="normal", font=dict(size=14),
                                      bordercolor="gray",   # Border color
                                      borderwidth=1,         # Border width in pixels
                                    ),
                          plot_bgcolor="white",  # Set background to white
                          paper_bgcolor="white" # Set paper background to white
                        )
        
        fig.update_xaxes(range=[0, 1])
        fig.add_trace(bar_trace, row=2, col=1)

        if draw_averages:
            avg_accuracy = bin_data["avg_accuracy"].item()
            avg_confidence = bin_data["avg_confidence"].item()
            acc_line = go.Scatter(x=[avg_accuracy, avg_accuracy], y=[0, np.min(counts)], mode='lines', line=dict(color='black', width=3), name="Avg. Overall Accuracy")
            conf_line = go.Scatter(x=[avg_confidence, avg_confidence], y=[0, np.min(counts)], mode='lines', line=dict(color='#444', width=3, dash='dot'), name="Avg. confidence")
            fig.add_trace(acc_line, row=2, col=1)
            fig.add_trace(conf_line, row=2, col=1)
            # Update layout properties
            fig.update_layout(legend=dict(title="Legend", orientation="h", x=0.5, y=-0.2, traceorder="normal"))

    def _reliability_diagram_subplot(self, fig, bin_data, draw_ece=True, draw_bin_importance=False, title="Reliability Diagram", xlabel="Confidence", ylabel="Expected Accuracy", show_axis_titles=True):
        """Draws a reliability diagram into a subplot."""
        accuracies = bin_data["accuracies"]
        confidences = bin_data["confidences"]
        counts = bin_data["counts"]
        bins = bin_data["bins"]

        bin_size = 1.0 / len(counts)
        positions = bins[:-1] + bin_size / 2.0

        widths = bin_size
        alphas = 0.3
        min_count = np.min(counts)
        max_count = np.max(counts)
        normalized_counts = (counts - min_count) / (max_count - min_count)

        if draw_bin_importance == "alpha":
            alphas = 0.2 + 0.8 * normalized_counts
        elif draw_bin_importance == "width":
            widths = 0.1 * bin_size + 0.9 * bin_size * normalized_counts

        # Initialize bar traces for gap and accuracy
        gap_trace = go.Bar(x=positions, y=np.abs(accuracies - confidences), base=np.minimum(accuracies, confidences), width=widths,
                           marker=dict(color=f"rgba(240, 60, 60, {alphas})", line=dict(color=f"rgba(240, 60, 60, {alphas})", width=1)), name="Gap")
        acc_trace = go.Bar(x=positions, y=np.zeros_like(accuracies), width=widths, base=accuracies, name="Accuracy",
                           marker=dict(color='black', line=dict(color='black', width=3)), opacity=1.0)
        # Create layout
        layout = go.Layout(title=title,
                           xaxis=dict(title=xlabel if show_axis_titles else None,  # xlabel
                                      range=[0, 1],
                                      showline=True, linewidth=1, linecolor="black", mirror=True,
                                      showgrid=True, gridcolor="lightgray",
                           ),
                           yaxis=dict(
                               title="" if show_axis_titles else None,  # ylabel
                               range=[0, 1],
                               showline=True, linewidth=1, linecolor="black", mirror=True,
                               showgrid=True, gridcolor="lightgray",
                               title_font=dict(size=18),  # Increase ylabel font size
                               tickfont=dict(size=14)  # Increase tick font size for y-axis
                            ),  
                           showlegend=True, barmode='overlay',
                           legend=dict(title="", x=0, y=1, traceorder="normal", font=dict(size=14),
                                       bordercolor="gray",   # Border color
                                        borderwidth=1,         # Border width in pixels
                            ), # orientation="h" # Legend at top left
                           plot_bgcolor="white",  # Set background to white
                           paper_bgcolor="white" # Set paper background to white
                        )

        fig.update_layout(layout)
        fig.add_trace(gap_trace, row=1, col=1)
        fig.add_trace(acc_trace, row=1, col=1)
        # Plot Ideal calibration
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"), row=1, col=1)
        # Add ECE text
        if draw_ece:
            # ece = bin_data["expected_calibration_error"] * 100   # ece:.2f in annotation if ece * 100
            ece = bin_data["expected_calibration_error"]
            fig.add_annotation(text=f"ECE={ece:.4f}", x=0.98, y=0.02, xanchor="right", yanchor="bottom", font=dict(color="black"), showarrow=False)

    def _reliability_diagram_combined(self, bin_data, draw_ece, draw_bin_importance, draw_averages, title, figsize, return_fig):
        """Draws a reliability diagram and confidence histogram"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7, 0.3])
        self._reliability_diagram_subplot(fig, bin_data, draw_ece, draw_bin_importance, title=title, xlabel="")
        # Draw the confidence histogram upside down.
        # orig_counts = bin_data["counts"]
        # bin_data["counts"] = -bin_data["counts"]
        self._confidence_histogram_subplot(fig, bin_data, draw_averages, title="")
        # bin_data["counts"] = orig_counts
        # Also change negative ticks to positive paranthesis ticks for the upside-down histogram.
        y_axis_layout = fig['layout']['yaxis2']
        y_axis_layout.tickformat = "("
        fig.update_layout(yaxis2=y_axis_layout)
        fig.update_layout(
            height=700,  # Set the height based on figsize  700
            width=500,  # Set the width based on figsize  500
        )
        if return_fig:
            return fig

    def reliability_diagram(self, bin_data, draw_ece=True, draw_bin_importance=True, draw_averages=False, title="Reliability Diagram", figsize=(6, 6), return_fig=True):
        """Draws a reliability diagram and confidence histogram in a single plot.
        First, the model's predictions are divided up into bins based on their
        confidence scores.

        The reliability diagram shows the gap between average accuracy and average
        confidence in each bin. These are the red bars.

        The black line is the accuracy, the other end of the bar is the confidence.

        Ideally, there is no gap and the black line is on the dotted diagonal.
        In that case, the model is properly calibrated and we can interpret the
        confidence scores as probabilities.

        The confidence histogram visualizes how many examples are in each bin.
        This is useful for judging how much each bin contributes to the calibration
        error.

        The confidence histogram also shows the overall accuracy and confidence.
        The closer these two lines are together, the better the calibration.

        The ECE or Expected Calibration Error is a summary statistic that gives the
        difference in expectation between confidence and accuracy. In other words,
        it's a weighted average of the gaps across all bins. A lower ECE is better.

        Arguments:
            true_labels: the true labels for the test examples
            pred_labels: the predicted labels for the test examples
            confidences: the predicted confidences for the test examples
            num_bins: number of bins
            draw_ece: whether to include the Expected Calibration Error
            draw_bin_importance: whether to represent how much each bin contributes
                to the total accuracy: False, "alpha", "widths"
            draw_averages: whether to draw the overall accuracy and confidence in
                the confidence histogram
            title: optional title for the plot
            figsize: setting for matplotlib; height is ignored
            return_fig: if True, returns the matplotlib Figure object
            Ref: https://github.com/hollance/reliability-diagrams/blob/master/README.markdown
        """
        for k, v in bin_data.items():
            bin_data[k] = self.ensure_numpy(v)
        print(f"bin_data: {bin_data}")
        return self._reliability_diagram_combined(bin_data, draw_ece, draw_bin_importance, draw_averages, title, figsize=figsize, return_fig=return_fig)


class ECEIndividual(MulticlassCalibrationError):
    """
    This function is just the original MulticlassCalibration error from torch metrics.
    Just calling forward function for each image in the dataloader gives ece for that image.
    Also, resetting state after every image so the internal state is not updated with the previous states.
    This makes sure to get ece metric for each image

    Adding reset functionality to internal states of original MulticlassCalibrationError.
    """
    def __init__(self, num_classes: int, **kwargs: Any,) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        
    def forward(self, probs: Tensor, labels: Tensor) -> Tensor:
        """
        Computes ECE for a single image by:
        - Adding the current image's data to the state
        - Computing the metric
        - Resetting the state after each computation
        """
        # Add the current batch (single image's data) to the internal state
        self.update(probs, labels)
        # Compute the ECE for this image
        ece_value = super().compute()
        # Reset the internal state to ensure no carry-over
        self.reset()
        return ece_value

    def reset(self):
        """Resets the internal states to ensure no carry-over between images."""
        print("INFO: Resetting the internal states of individual ECE")
        self.confidences = []
        self.accuracies = []
        self._computed = None  # If there are any cached results


class MisclassifiedConfidenceHistogram(Metric):
    def __init__(self, num_classes, num_bins=10, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.num_bins = num_bins

        # Define bin edges and initialize histogram accumulator
        self.register_buffer("bin_edges", torch.linspace(0, 1, num_bins + 1))
        self.register_buffer("histogram", torch.zeros(num_bins, dtype=torch.float))

    def update(self, confidences: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update the histogram with new data.
        
        Args:
            confidences (torch.Tensor): Confidence scores (probabilities) for each pixel.
            preds (torch.Tensor): Predicted class labels for each pixel.
            targets (torch.Tensor): Ground truth class labels for each pixel.
        """
        # Ensure input dimensions match
        assert confidences.shape == preds.shape == targets.shape, \
            "Confidences, predictions, and targets must have the same shape."

        # Identify misclassified pixels
        targets = targets.to(torch.int64)
        preds = preds.to(torch.int64)
        ignore_index = 255
        valid_mask = targets != ignore_index
        misclassified_mask = (preds != targets) & valid_mask

        # Gather confidences of misclassified pixels
        misclassified_confidences = confidences[misclassified_mask]

        # Update histogram
        # hist = torch.histc(misclassified_confidences, bins=self.num_bins, min=0.0, max=1.0)
        # self.histogram += hist
        if misclassified_confidences.numel() > 0:  # Ensure it's not empty
             # Move to CPU for histogram calculation
            hist, _ = torch.histogram(misclassified_confidences.cpu(), bins=self.bin_edges.cpu())
            self.histogram += hist.to(self.histogram.device)
        else:
            print("[INFO]: No misclassified pixels")

    def compute(self):
        """
        Compute the final histogram.
        
        Returns:
            histogram (torch.Tensor): Normalized histogram of misclassified pixel confidences.
            bin_edges (torch.Tensor): Bin edges for the histogram.
        """
        return self.histogram, self.bin_edges

    def reset(self):
        """
        Reset the metric.
        """
        self.histogram.zero_()
