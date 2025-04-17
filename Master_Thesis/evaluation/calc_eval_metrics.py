# FIXME : Handle edl unc and beliefs while plotting

import os
import pickle
import gc
import json

import torch
from torch.utils.data import DataLoader

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, JaccardIndex

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import plotly.graph_objects as go

import wandb
from tqdm import tqdm

from evaluation.evaluation_metrics import ClassWiseFrequency, ClassWiseUncertainty, CustomMultiClassCalibrationError, ECEIndividual, NegativeLogLikelihood, compute_aurrc_selection_thresholds, compute_aurrc_risks_rejections, MisclassifiedConfidenceHistogram # area_under_risk_rejection_rate_curve
from evaluation.evaluation_utils import get_labels_data, save_per_img_results, LoadOutputsDataset, LoadTwoOutputDataset, LoadOODIDDataset, LoadMisclassificationDataset, log_eval_metrics
from evaluation.evaluation_ood_metrics import OODMetrics

from utils.plot_utils import plot_aupr_curve, plot_entropy_distribution, plot_per_class_metrics, plot_risks_curves, plot_roc_curve, plot_per_img_results, plot_roc_curve_multiple


class TestMetrics:
    def __init__(self, args, run, gpu_id):
        self.is_distributed = args.is_distributed
        # Construct the final plots save dir
        model_name = args.model_path.split('/')[-1].replace('model_', '').replace('.pt', '')
        self.plots_save_dir = model_name
        
        self.device = gpu_id
        self.dataset_name = args.eval_dataset_name
        self.id2label, self.label2id, self.labels = get_labels_data(dataset_name=self.dataset_name)
        self.num_classes = len(self.id2label)
        self.ignore_index = args.ignore_index  # 255

        self.calc_seg_metrics = args.calc_seg_metrics
        if self.calc_seg_metrics:
            self.output_dir = args.tests_save_dir  # can be anyone of pathtomodelnames/ Original, MCDInference, OOD_<Datasetname>, OOD_<Datasetname>/MCDInference, Robustness_<test_name>

        self.calc_ece = args.calc_ece
        if self.calc_ece:
            self.ece_base_dir = args.ece_base_dir
            self.ece_bins = args.ece_bins

        self.calc_aurrc = args.calc_aurrc
        if self.calc_aurrc:
            self.aurrc_base_dir = args.aurrc_base_dir
            self.aurrc_bins = args.aurrc_bins

        self.calc_ood_metrics = args.calc_ood_metrics
        self.is_ood_multiple = args.is_ood_multiple
        if self.calc_ood_metrics:
            self.ood_name = args.ood_name.lower()
            self.recall_level = args.recall_level
            if self.is_ood_multiple:
                self.multiple_ood_dirs = args.multiple_ood_dirs
            else:
                self.ood_dir = args.ood_dir
                assert self.ood_name == os.path.basename(self.ood_dir).split("OOD_")[1].lower(), f"Given ood_name: {self.ood_name} doesnt match with ood_dir: {self.ood_dir}"
        
        # Here
        self.calc_misclassification = args.calc_misclassification
        if self.calc_misclassification:
            self.misclassification_dir = args.misclassification_dir
            self.misclassification_name = args.misclassification_name.lower()
            # Check the paths
            path_base_name = os.path.basename(self.misclassification_dir)
            self.is_miscclass_ood = False
            if self.misclassification_name == "cityscapes":
                assert path_base_name == "Original", f"Given misclassification_name: {self.misclassification_name} doesnt match with misclassification_dir: {self.misclassification_dir}"
            else:
                self.is_miscclass_ood = True
                assert self.misclassification_name == path_base_name.split("OOD_")[1].lower(), f"Given misclassification_name: {self.misclassification_name} doesnt match with misclassification_dir: {self.misclassification_dir}"
            self.misclass_recall_level = args.misclass_recall_level

        self.plot_id_ood_distributions = args.plot_id_ood_distributions
        if self.plot_id_ood_distributions:
            self.baseline_ood_dir = args.baseline_ood_dir
            self.improv_ood_dir = args.improv_ood_dir
            self.baseline_mdl_title = args.baseline_mdl_title
            self.improv_mdl_title = args.improv_mdl_title

        # Initialize datasets and metrics
        self._init_datasets()
        self._init_metrics()

        if run is not None:
            # run.config["classes"] = len(self.id2label)
            self.run = run

    def _init_datasets(self) -> None:
        if self.calc_seg_metrics:
            self.pred_var_exists = os.path.exists(os.path.join(self.output_dir, "pred_var"))
            self.edl_belief_exists = os.path.exists(os.path.join(self.output_dir, "edl_uncertainty")) # and os.path.exists(os.path.join(self.output_dir, "edl_beliefs"))
            self.outputs_dataset = LoadOutputsDataset(base_dir=self.output_dir, pred_var_exists=self.pred_var_exists, edl_belief_exists=self.edl_belief_exists)

        if self.calc_ece:
            self.ece_dataset = LoadTwoOutputDataset(base_dir=self.ece_base_dir, dir1_name="probs", dir2_name="labels")

        if self.calc_aurrc:
            self.aurrc_selection_threshold_dataset = LoadTwoOutputDataset(base_dir=self.aurrc_base_dir, dir1_name="probs", dir2_name="labels")
            self.aurrc_risks_rejections_dataset = LoadTwoOutputDataset(base_dir=self.aurrc_base_dir, dir1_name="preds", dir2_name="labels")

        self.edl_unc_exists = False
        if self.calc_ood_metrics:
            if self.is_ood_multiple:
                self.multiple_ood_datasets =[]
                # Validate that multiple_ood_dirs contains valid data
                if not self.multiple_ood_dirs:
                    print("[ERROR]: No OOD directories provided.")
                    return
                for ood_dir in self.multiple_ood_dirs:
                    print(ood_dir)
                    ood_name = ood_dir['name']
                    ood_path = ood_dir['path']
                    # Construct the file paths for max_pred_probs and labels
                    max_pred_probs_path = os.path.join(ood_path, "max_pred_probs")
                    ood_label_file_path = os.path.join(ood_path, "labels")
                    # Check if the required files exist
                    if not os.path.exists(max_pred_probs_path):
                        print(f"[ERROR]: max_pred_probs file does not exist for {ood_name} at {max_pred_probs_path}")
                        return
                    if not os.path.exists(ood_label_file_path):
                        print(f"[ERROR]: labels file does not exist for {ood_name} at {ood_label_file_path}")
                        return
                    try:
                        ood_dataset = LoadOODIDDataset(ood_conf_file=max_pred_probs_path, ood_label_file=ood_label_file_path)
                        # Append the OOD dataset to the list
                        self.multiple_ood_datasets.append({"name": ood_name, "dataset": ood_dataset})
                        print(f"Successfully loaded OOD dataset: {ood_name}")
                    except Exception as e:
                        print(f"Error loading OOD dataset {ood_name}: {e}")
            else:
                self.ood_dataset = LoadOODIDDataset(ood_conf_file=f"{self.ood_dir}/max_pred_probs", ood_label_file=f"{self.ood_dir}/labels")
                if os.path.exists(os.path.join(self.ood_dir, "edl_uncertainty")):
                    self.edl_unc_exists = True
                    self.ood_unc_dataset = LoadOODIDDataset(ood_conf_file=f"{self.ood_dir}/edl_uncertainty", ood_label_file=f"{self.ood_dir}/labels")

        if self.calc_misclassification:
            print("Calculating Misclassification")
            # self.misclassification_dataset = LoadMisclassificationDataset(conf_file=f"{self.misclassification_dir}/max_pred_probs", pred_file=f"{self.misclassification_dir}/preds", label_file=f"{self.misclassification_dir}/labels", is_ood=self.is_miscclass_ood)
            self.misclassification_dataset = LoadMisclassificationDataset(conf_file=f"{self.misclassification_dir}/max_pred_probs", pred_file=f"{self.misclassification_dir}/preds", label_file=f"{self.misclassification_dir}/labels")
        
        if self.plot_id_ood_distributions:
            # FIXME: Can We plot MCDpredvar and edl_unc too?
            self.ood_entropy_baseline_dir = f"{self.baseline_ood_dir}/entropies"
            self.ood_labels_dir = f"{self.baseline_ood_dir}/labels"
            self.ood_entropy_improv_baseline_dir = f"{self.improv_ood_dir}/entropies"

    def _init_metrics(self) -> None:
        # Metrics to log
        self.scalar_metrics_to_log = []  # ["mean_iou", "overall_accuracy", "ece", "mean_accuracy", "nll", "aurrc", "fpr_ood", "aupr_ood", "auroc_ood"]
        self.per_category_metrics_to_log = []  # ["per_category_iou", "per_category_accuracy", "per_category_entropy", "per_category_mcd_unc", "per_category_edl_belief", "per_category_edl_unc", "per_category_freq"]  # These are lists

        if self.calc_seg_metrics:
            # Seg Metrics
            metrics_collection = MetricCollection({
                "iou_per_category": JaccardIndex(task="multiclass", num_classes=self.num_classes, average="none", ignore_index=self.ignore_index),       # Per Category IoU
                "acc_per_category": Accuracy(task="multiclass", num_classes=self.num_classes, average="none", ignore_index=self.ignore_index),           # Per Category Accuracy
                "IoU_weighted": JaccardIndex(task="multiclass", num_classes=self.num_classes, average='weighted', ignore_index=self.ignore_index),       # mIoU
                "overall_acc_weighted": Accuracy(task="multiclass", num_classes=self.num_classes, average='weighted', ignore_index=self.ignore_index),   # overall_Accuracy
            })
            self.seg_metrics = metrics_collection.clone(prefix="test_")
            self.nll = NegativeLogLikelihood(ignore_index=self.ignore_index)                                                                 # Negative Log Likelihood
            self.nll_per_img = NegativeLogLikelihood(ignore_index=self.ignore_index)                                                                 # # Negative Log Likelihood Per Image

            self.scalar_metrics_to_log.extend(["mean_iou", "overall_accuracy", "mean_accuracy", "nll"])

            # Per Class Metrics
            self.class_wise_entropy = ClassWiseUncertainty(num_classes=self.num_classes, ignore_index=self.ignore_index)                  # Per Category Entropy
            self.class_wise_freq = ClassWiseFrequency(num_classes=self.num_classes, ignore_index=self.ignore_index)                       # Per Category Pixel Frequency
            self.per_category_metrics_to_log.extend(["per_category_iou", "per_category_accuracy", "per_category_entropy", "per_category_freq"])

            if self.pred_var_exists:
                self.class_wise_uncertainty = ClassWiseUncertainty(num_classes=self.num_classes, ignore_index=self.ignore_index)              # Per Category Uncertainty
                self.per_category_metrics_to_log.extend(["per_category_mcd_unc"])

            if self.edl_belief_exists:
                self.class_wise_edl_unc = ClassWiseUncertainty(num_classes=self.num_classes, ignore_index=self.ignore_index)                  # Per Category Uncertainty for EDL Models
                # self.class_wise_edl_belief = ClassWiseUncertainty(num_classes=self.num_classes, ignore_index=self.ignore_index)               # Per Category Belief for EDL Models
                self.per_category_metrics_to_log.extend(["per_category_edl_unc"])  # per_category_edl_belief

            # Per Image Metrics
            self.iou_per_img = JaccardIndex(task="multiclass", num_classes=self.num_classes, average='weighted', ignore_index=self.ignore_index)  # IoU per image
            self.acc_per_img = Accuracy(task="multiclass", num_classes=self.num_classes, average='weighted', ignore_index=self.ignore_index)      # Accuracy per image

        # ECE Metric
        if self.calc_ece:
            self.ece = CustomMultiClassCalibrationError(num_classes=self.num_classes, n_bins=self.ece_bins, norm="l1", ignore_index=self.ignore_index)  # ECE
            self.scalar_metrics_to_log.extend(["ece"])
            # Per Image Metric
            self.ece_per_img = ECEIndividual(num_classes=self.num_classes, n_bins=self.ece_bins, norm="l1", ignore_index=self.ignore_index)             # ECE per image
        
        # OOD Metric
        if self.calc_ood_metrics:
            self.ood_metrics = OODMetrics(recall_level=self.recall_level, pos_label=1, ignore_index=self.ignore_index)                                      # AUROC, AUPR, FPR@thresh
            self.scalar_metrics_to_log.extend([f"fpr_at_tpr_{self.ood_name}_ood", f"aupr_in_{self.ood_name}_ood", f"aupr_out_{self.ood_name}_ood", f"auroc_{self.ood_name}_ood"])
            if self.edl_unc_exists:
                self.scalar_metrics_to_log.extend([f"fpr_at_tpr_{self.ood_name}_unc_ood", f"aupr_in_{self.ood_name}_unc_ood", f"aupr_out_{self.ood_name}_unc_ood", f"auroc_{self.ood_name}_unc_ood"])

        # Miscclassification, we use OOD Metrics also for misclassification
        if self.calc_misclassification:
            self.misclassification_metrics = MisclassifiedConfidenceHistogram(num_classes=self.num_classes, num_bins=10)
            # self.misclassification_metrics = OODMetrics(recall_level=self.misclass_recall_level, pos_label=1, ignore_index=self.ignore_index)                                      # AUROC, AUPR, FPR@thresh
            # self.scalar_metrics_to_log.extend([f"fpr_at_tpr_{self.misclassification_name}_misclass", f"aupr_in_{self.misclassification_name}_misclass", f"aupr_out_{self.misclassification_name}_misclass", f"auroc_{self.misclassification_name}_misclass"])
     
        # AURRC
        if self.calc_aurrc:
            self.scalar_metrics_to_log.extend(["aurrc"])

    def ece_evaluation(self):
        self.ece.to(self.device)
        # Load the the probs and labels from the outputs dataset.  We are using separate method bcz we dont want to fill up the memory with other data.
        ece_dataloader = DataLoader(self.ece_dataset, batch_size=1, pin_memory=True, shuffle=False)
        # Note: Ece internal states in the Metric class are saving whole data
        for i, data_dict in enumerate(tqdm(ece_dataloader)):
            probs = data_dict["probs"].squeeze(dim=1).to(self.device)
            labels = data_dict["labels"].squeeze(dim=1).to(self.device)
            self.ece.update(probs, labels)
            del data_dict
            
        bin_data = self.ece.compute()
        ece_fig = self.ece.reliability_diagram(bin_data)
        ece_value = self._extract_ece_value(bin_data)
        self.ece.reset()
        del probs, labels
        return ece_value, ece_fig

    def _extract_ece_value(self, bin_data) -> float:
        if self.ece.norm == "l1":
            return bin_data["expected_calibration_error"].item()
        elif self.ece.norm == "max":
            return bin_data["max_calibration_error"].item()
        elif self.ece.norm == "l2":
            return bin_data["l2_calibration_error"].item()
        else:
            raise ValueError("Invalid normalization method.")

    def get_aurrc_selection_thresholds(self):
        print("Running compute_aurrc_selection_thresholds")
        # Load the confidence scores and actual labels from the whole dataset with from the outputs of the given path
        aurrc_dataloader = DataLoader(self.aurrc_selection_threshold_dataset, batch_size=1, pin_memory=True, shuffle=False)
        
        all_preds = []
        all_selection_scores = []
        indices_mask = []
        
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(aurrc_dataloader)):
                probs = data_dict["probs"].squeeze(dim=1).cpu()
                selection_scores, preds = torch.max(probs, dim=1)
                labels = data_dict["labels"].squeeze(dim=1).cpu()
                mask = labels != self.ignore_index
                
                selection_scores = selection_scores.cpu()
                preds = preds.cpu()
                mask = mask.cpu()
                
                all_preds.append(preds)
                all_selection_scores.append(selection_scores)
                indices_mask.append(mask)

                del data_dict, probs, preds, selection_scores, labels, mask
                torch.cuda.empty_cache()  # Free up GPU memory
                gc.collect()

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_selection_scores = torch.cat(all_selection_scores, dim=0)
        indices_mask = torch.cat(indices_mask, dim=0)
        
        # Filter out the ignore index
        all_preds = all_preds[indices_mask].numpy()
        all_selection_scores = all_selection_scores[indices_mask].numpy()
       
        print(f"all_preds: {all_preds.shape}, all_selection_scores: {all_selection_scores.shape}, indices_mask: {indices_mask.shape}")
        
        selection_thresholds, bin_indices = compute_aurrc_selection_thresholds(all_preds, all_selection_scores, num_bins=self.aurrc_bins)

        del all_preds, all_selection_scores, indices_mask
        torch.cuda.empty_cache()  # Free up GPU memory
        gc.collect()
        return selection_thresholds, bin_indices
    
    def get_aurrc_risks_rejections(self, bin_indices):
        print("Running get_aurrc_risks_rejections")
        
        # Load the confidence scores and actual labels from the whole dataset with from the outputs of the given path
        aurrc_dataloader = DataLoader(self.aurrc_risks_rejections_dataset, batch_size=1, pin_memory=True, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(aurrc_dataloader)):
                preds = data_dict["preds"].squeeze(dim=1).cpu()  # .long()
                labels = data_dict["labels"].squeeze(dim=1).cpu()  #.long()
                mask = labels != self.ignore_index
                preds = preds[mask]
                labels = labels[mask]
                all_preds.append(preds)
                all_labels.append(labels)

                del data_dict, preds, labels, mask
                torch.cuda.empty_cache()  # Free up GPU memory
                # gc.collect()
        
        print("[INFO]: Concatentating in risk rejections")
        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
       
        print(f"all_preds: {all_preds.shape}, all_labels: {all_labels.shape}")
        
        aurrrc, rejection_rates, risks =  compute_aurrc_risks_rejections(all_preds, all_labels, bin_indices, risk_func=accuracy_score)
        del all_labels, all_preds
        torch.cuda.empty_cache()  # Free up GPU memory
        return aurrrc, rejection_rates, risks
        
    def aurrc_evaluation(self):
        print("[INFO]: AURRC Evaluation...")
        selection_thresholds, bin_indices = self.get_aurrc_selection_thresholds()
        aurrrc, rejection_rates, risks = self.get_aurrc_risks_rejections(bin_indices)
        # Plots 
        try:
            risks_rejections_fig = plot_risks_curves(x_vals=rejection_rates, y_vals=risks, aurrc=aurrrc, x_title="Rejection Rates", y_title="Risks")
            risks_selection_thresholds_fig = plot_risks_curves(x_vals=selection_thresholds, y_vals=risks, aurrc=aurrrc, x_title="Selection Thresholds", y_title="Risks")
        except Exception as e:
            print("[WARN]: Failed to plot the rejection curves")
            risks_rejections_fig = None
            risks_selection_thresholds_fig = None
        
        del bin_indices, selection_thresholds, rejection_rates, risks
        torch.cuda.empty_cache()
        gc.collect()
        return aurrrc, risks_rejections_fig, risks_selection_thresholds_fig

    def ood_evaluation(self):
        print("[INFO]: OOD Evaluation...")
        
        self.ood_metrics.to(self.device)
        # Load the confidence scores and ood labels from the whole dataset with from the outputs of the given path
        self.ood_dataloader = DataLoader(self.ood_dataset, batch_size=1, pin_memory=True, shuffle=False)
        for i, data_dict in enumerate(tqdm(self.ood_dataloader)):
            conf_score = data_dict["scores"].squeeze(dim=1).to(self.device)
            target = data_dict["label"].squeeze(dim=1).to(self.device)
            # Update OOD Metrics
            self.ood_metrics.update(-conf_score, target)
            del data_dict
        # Compute OOD Metrics
        fpr_at_recall_level, threshold_at_recall_level, aupr_in, aupr_out, auroc, roc_curve_data, aupr_in_curve_data, aupr_out_curve_data = self.ood_metrics.compute()
        # Plot each curve
        fig_roc = plot_roc_curve(roc_curve_data, auroc, fpr_at_recall_level, self.recall_level)
        # fig_aupr_in = plot_aupr_curve(aupr_in_curve_data, aupr_in, title="in")
        # fig_aupr_out = plot_aupr_curve(aupr_out_curve_data, aupr_out, title="out")
        self.ood_metrics.reset()
        torch.cuda.empty_cache()
        return fpr_at_recall_level, aupr_in, aupr_out, auroc, fig_roc #, fig_aupr_in, fig_aupr_out
    
    def ood_evaluation_multiple(self, dataloader_list):
        """
        Perform OOD evaluation for multiple dataloaders sequentially, saving intermediate results.
        Args:
            dataloader_list: List of dataloaders to evaluate.
        Returns:
            Combined ROC curve figure for all dataloaders.
        """
        print("[INFO]: OOD Evaluation for multiple dataloaders...")
        file_names = []
        # file_names = ["roc_result_0_Baseline", "roc_result_1_MCD", "roc_result_2_OHEM", "roc_result_3_EDL"]

        for idx, loader_dict in enumerate(dataloader_list):
            ood_dataset = loader_dict["dataset"]
            label = loader_dict["name"]
            print(f"[INFO]: Evaluating for {label}...")
            
            self.ood_metrics.to(self.device)
            ood_dataloader = DataLoader(ood_dataset, batch_size=1, pin_memory=True, shuffle=False)
            
            for i, data_dict in enumerate(tqdm(ood_dataloader)):
                conf_score = data_dict["scores"].squeeze(dim=1).to(self.device)
                target = data_dict["label"].squeeze(dim=1).to(self.device)
                self.ood_metrics.update(-conf_score, target)
                del data_dict  # Free memory

            fpr_at_recall_level, _, _, _, auroc, roc_curve_data, _, _ = self.ood_metrics.compute()

            # Save intermediate results to file
            name = f"roc_result_{idx}_{label}"
            file_names.append(name)
            result_file = f"{name}.pkl"
            try:
                with open(result_file, 'wb') as f:
                    pickle.dump((roc_curve_data, auroc, fpr_at_recall_level, label), f)
                    print(f"[INFO]: Saved results for {label} to {result_file}")
            except Exception as e:
                print(f"[ERROR]: Failed to save file: {name}.pkl")
                
            # Clear metrics and memory
            self.ood_metrics.reset()
            torch.cuda.empty_cache()
        fig = self.plot_multiple_roc_curve(file_names)
        return fig
    
    def plot_multiple_roc_curve(self, file_names):
        print("[INFO]: Plotting the Multiple ROC Curves")
        
        # # Load the saved Plotly figure from a JSON file
        with open('lostmcd_ohem_edl_081range.json', 'r') as f:
            fig_data = json.load(f)
        # Create a Plotly Figure object from the JSON data
        fig = go.Figure(fig_data)
        
        ## Combine results into one figure
        # fig = go.Figure()
        for result_file in file_names:
            with open(f"{result_file}.pkl", 'rb') as f:
                print(f"[INFO]: Opening the result file {result_file}")
                roc_curve_data, auroc, fpr_at_recall_level, label = pickle.load(f)
            
                fpr, tpr = roc_curve_data[0], roc_curve_data[1]
                # fig.add_trace(
                #     go.Scattergl(
                #         x=fpr,
                #         y=tpr,
                #         mode='lines',
                #         name=f'{label}: AUROC={auroc:.3f}, FPR@95={fpr_at_recall_level:.3f}',
                #         showlegend=True,
                #     )
                # )
                fig.add_trace(
                    go.Scattergl(
                        x=[fpr_at_recall_level],
                        y=[self.recall_level],
                        mode='markers',
                        # marker=dict(color='red', size=5),
                        # showlegend=False,
                        marker=dict(color='black', size=8, symbol='star'),
                        name=f'{label}: AUROC={auroc:.3f}, FPR@95={fpr_at_recall_level:.3f}',
                        showlegend=True,
                    )
                )

        # #Add Random Classifier baseline
        # fig.add_trace(
        #     go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash'))
        # )

        # Update layout
        fig.update_layout(
            title="",
            xaxis=dict(
                title="False Positive Rate",
                showline=True, linewidth=1, linecolor="black", mirror=True,
                showgrid=True, gridcolor="lightgray",
            ),
            yaxis=dict(
                title="True Positive Rate",
                showline=True, linewidth=1, linecolor="black", mirror=True,
                showgrid=True, gridcolor="lightgray",
                title_font=dict(size=18),
                tickfont=dict(size=14),
                range=[0.8, 1]  # [0.8, 1]
            ),
            showlegend=True,
            legend=dict(
                title="",  # <b>Legend</b>
                x=0.99, y=0.01,
                xanchor="right", yanchor="bottom",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=12)
            ),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        # # # Optionally delete saved result files
        # # for result_file in file_names:
        # #     os.remove(result_file.pkl)
        
        # Save the original figure to JSON
        with open('lostmcd_ohem_edl_081range_up.json', 'w') as f:
            json.dump(fig.to_dict(), f)
        return fig

    def ood_unc_evaluation(self):
        print("[INFO]: OOD Uncertainty Evaluation...")
        
        self.ood_metrics.to(self.device)
        # Load the confidence scores and ood labels from the whole dataset with from the outputs of the given path
        self.ood_dataloader = DataLoader(self.ood_unc_dataset, batch_size=1, pin_memory=True, shuffle=False)
        for i, data_dict in enumerate(tqdm(self.ood_dataloader)):
            unc_score = data_dict["scores"].squeeze(dim=1).to(self.device)
            target = data_dict["label"].squeeze(dim=1).to(self.device)
            
            # Update OOD Metrics
            self.ood_metrics.update(unc_score, target)
            del data_dict
        # Compute OOD Metrics
        fpr_at_recall_level, threshold_at_recall_level, aupr_in, aupr_out, auroc, roc_curve_data, aupr_in_curve_data, aupr_out_curve_data = self.ood_metrics.compute()
        # Plot each curve
        # fig_roc = plot_roc_curve(roc_curve_data, auroc, fpr_at_recall_level, self.recall_level)
        # fig_aupr_in = plot_aupr_curve(aupr_in_curve_data, aupr_in, title="in")
        # fig_aupr_out = plot_aupr_curve(aupr_out_curve_data, aupr_out, title="out")
        self.ood_metrics.reset()
        return fpr_at_recall_level, aupr_in, aupr_out, auroc # , fig_roc
    
    def misclassification_evaluation(self):
        print("[INFO]: Misclassification Evaluation...")
        self.misclassification_metrics.to(self.device)
        # Load the confidence scores and ood labels from the whole dataset with from the outputs of the given path
        self.misclassification_dataloader = DataLoader(self.misclassification_dataset, batch_size=1, pin_memory=True, shuffle=False)
        for i, data_dict in enumerate(tqdm(self.misclassification_dataloader)):
            conf_score = data_dict["scores"].squeeze(dim=1).to(self.device)
            target = data_dict["label"].squeeze(dim=1).to(self.device)
            pred = data_dict["pred"].squeeze(dim=1).to(self.device)
            # Update OOD Metrics
            self.misclassification_metrics.update(conf_score, pred, target)
            del data_dict  # , conf_score, target
        print("computing")
        # Compute histogram
        misclass_histogram, bin_edges = self.misclassification_metrics.compute()
        # Plot histogram
        bin_width = bin_edges[1] - bin_edges[0]
        # Create the histogram with Plotly
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=bin_edges[:-1].cpu(),
                y=misclass_histogram.cpu(),
                width=bin_width.item(),
                marker=dict(color='blue', line=dict(color='black', width=1)),
                opacity=0.7,
                name="Misclassified Pixels"
            )
        )

        # Customize the layout
        fig.update_layout(
            title="",
            xaxis=dict(title="Confidence of Misclassified Pixels",
                       showline=True, linewidth=1, linecolor="black", mirror=True,
                       title_font=dict(size=18),
                       tickfont=dict(size=14)),
            yaxis=dict(title="Number of Pixels", 
                       showline=True, linewidth=1, linecolor="black", mirror=True,
                       showgrid=True, gridcolor="lightgray",
                       title_font=dict(size=18),
                       tickfont=dict(size=14)),
            bargap=0.1,  # Add slight spacing between bars
            plot_bgcolor="white",  # Set background to white
            paper_bgcolor="white"  # Set paper background to white
        )

        self.misclassification_metrics.reset()
        return fig

    def segmentation_evaluation(self):
        # Seg Metrics
        self.seg_metrics.to(self.device)
        self.class_wise_entropy.to(self.device)
        self.class_wise_freq.to(self.device)
        if self.pred_var_exists:
            self.class_wise_uncertainty.to(self.device)
        if self.edl_belief_exists:
            self.class_wise_edl_unc.to(self.device)
            # self.class_wise_edl_belief.to(self.device)
        self.nll.to(self.device)
        self.nll_per_img.to(self.device)
        if self.calc_ece:
            self.ece_per_img.to(self.device)
        self.iou_per_img.to(self.device)
        self.acc_per_img.to(self.device)

        # For Saving the Test Metrics to a text file
        imgs_IoUs, imgs_accs, imgs_ece, imgs_nlls, imgs_names   = [], [], [], [], []

        # Dataloader
        seg_dataloader = DataLoader(self.outputs_dataset, batch_size=1, pin_memory=True, shuffle=False)
        for idx, data_dict in enumerate(tqdm(seg_dataloader)):
            preds = data_dict["preds"].squeeze(dim=1).to(self.device)
            labels = data_dict["labels"].squeeze(dim=1).to(self.device)
            probs = data_dict["probs"].squeeze(dim=1).to(self.device)

            # Seg Metrics
            print(f"Preds {preds.shape}, Labels: {labels.long().shape}, Probs: {probs.shape}")
            seg_metrics = self.seg_metrics.update(probs, labels.long())  # FIXME: check the probs size

            # Negative Log Likelihood
            self.nll.update(probs, labels.long())
            self.nll_per_img.update(probs, labels.long())
            nll_per_img = self.nll_per_img.compute()
            self.nll_per_img.reset()

            # Per Class Entropy
            entropies = data_dict["entropies"].squeeze(dim=1).to(self.device)
            self.class_wise_entropy.update(preds.long(), labels.long(), entropies)

            # Per Class Predictive Variance
            if self.pred_var_exists:
                pred_var = data_dict["pred_var"].squeeze(dim=1).to(self.device)
                self.class_wise_uncertainty.update(preds.long(), labels.long(), pred_var)

            # Per Class EDL Belief and Uncertainty
            if self.edl_belief_exists:
                # edl_belief = data_dict["edl_belief"].squeeze(dim=1).to(self.device)
                # self.class_wise_edl_belief.update(preds, labels.long(), edl_belief)
                edl_unc = data_dict["edl_uncertainty"].squeeze(dim=1).to(self.device)
                self.class_wise_edl_unc.update(preds.long(), labels.long(), edl_unc.squeeze(dim=1))

            # Per Class Pixel frequency
            self.class_wise_freq.update(preds.long(), labels.long())

            # Save the per image metrics for statistical significance testing, make sure you have single image in bs
            # Ece Per Image
            if self.calc_ece:
                ece_per_image = self.ece_per_img(probs, labels.long())
                # self.ece_per_img.reset()

            # IoU Per Image
            iou_per_img = self.iou_per_img(preds.long(), labels.long())
            self.iou_per_img.reset()

            # Accuracy Per Image
            acc_per_image = self.acc_per_img(preds.long(), labels.long())
            self.acc_per_img.reset()

            imgs_IoUs.append(iou_per_img.item())
            imgs_accs.append(acc_per_image.item())
            imgs_nlls.append(nll_per_img.item())
            if self.calc_ece:
                imgs_ece.append(ece_per_image.item())
            imgs_names.append(data_dict["image_name"][0])

            del data_dict

        # Compute all Metrics
        class_wise_uncertainty = self.class_wise_uncertainty.compute() if self.pred_var_exists else None
        # class_wise_edl_belief = self.class_wise_edl_belief.compute() if self.edl_belief_exists else None
        class_wise_edl_unc = self.class_wise_edl_unc.compute() if self.edl_belief_exists else None
        class_wise_entropy = self.class_wise_entropy.compute()
        class_wise_pixel_freq = self.class_wise_freq.compute()

        seg_metrics = self.seg_metrics.compute()
        seg_nll = self.nll.compute()

        mean_acc = torch.nanmean(seg_metrics["test_acc_per_category"])
        final_seg_metrics = {
            "mean_iou": seg_metrics["test_IoU_weighted"].item(),
            "overall_accuracy": seg_metrics["test_overall_acc_weighted"].item(),
            "mean_accuracy": mean_acc.item(),
            "nll": seg_nll.item(),
            "per_category_iou": seg_metrics["test_iou_per_category"],
            "per_category_accuracy": seg_metrics["test_acc_per_category"],
            "per_category_entropy": class_wise_entropy,
            "per_category_mcd_unc": class_wise_uncertainty,
            # "per_category_edl_belief": class_wise_edl_belief,
            "per_category_edl_unc": class_wise_edl_unc,
            "per_category_freq": class_wise_pixel_freq,
        }

        # Reset the Metrics
        self.seg_metrics.reset()
        self.class_wise_entropy.reset()
        if self.pred_var_exists:
            self.class_wise_uncertainty.reset()
        if self.edl_belief_exists:
            # self.class_wise_edl_belief.reset()
            self.class_wise_edl_unc.reset()
        self.class_wise_freq.reset()
        self.nll.reset()

        # Save per img results to a text file
        # Extract the base directory
        base_dir = os.path.dirname(self.output_dir)
        # Construct the new path
        per_img_path = os.path.join(base_dir, "Per_Image_Results")
        os.makedirs(per_img_path, exist_ok=True)
        file_name = self.output_dir.split('/')[2]
        save_file_name = f"{per_img_path}/per_image_test_results_{file_name}.csv"
        save_per_img_results(imgs_IoUs, imgs_accs, imgs_ece, imgs_nlls, imgs_names, save_file_name)
        fig_per_imgs = plot_per_img_results(save_path=f"{per_img_path}/per_image_test_results_{file_name}.csv")
        # Calculate Mean and std of per image metrics
        mean_acc = np.mean(imgs_accs)
        std_acc = np.std(imgs_accs)

        mean_ece = np.mean(imgs_ece)
        std_ece = np.std(imgs_ece)

        mean_nll = np.mean(imgs_nlls)
        std_nll = np.std(imgs_nlls)
        
        mean_mIoU = np.mean(imgs_IoUs)
        std_mIoU = np.std(imgs_IoUs)
        
        # Add the calculated values to the dictionary
        final_seg_metrics['per_img_macc_std'] = f"{mean_acc:.4f} ± {std_acc:.4f}"
        final_seg_metrics['per_img_mece_std'] = f"{mean_ece:.4f} ± {std_ece:.4f}"
        final_seg_metrics['per_img_mnll_std'] = f"{mean_nll:.4f} ± {std_nll:.4f}"
        final_seg_metrics['per_img_mmiou_std'] = f"{mean_mIoU:.4f} ± {std_mIoU:.4f}"
        
        return final_seg_metrics, fig_per_imgs

    def main(self):
        summary_metrics = {}
        figs = {}
        # Step 1: Calculate AURRC
        if self.calc_aurrc:
            print("[INFO]: Calculating AURRC")
            aurrrc, risks_rejections_fig, risks_selection_thresholds_fig = self.aurrc_evaluation()
            summary_metrics["aurrc"] = aurrrc
            figs["fig_risks_rejections"] = risks_rejections_fig
            figs["fig_risks_selection_thresholds"] = risks_selection_thresholds_fig
        # Step 2: Calculate ECE
        if self.calc_ece:
            print("[INFO]: Calculating ECE")
            ece, ece_fig = self.ece_evaluation()
            summary_metrics["ece"] = ece
            figs["fig_ece"] = ece_fig
        # Step 3: Calculate OOD metrics
        if self.calc_ood_metrics:
            print("[INFO]: Calculating OOD")
            if self.is_ood_multiple:
                print("[INFO]: Getting OOD multiple figure")
                multiple_roc_fig = self.ood_evaluation_multiple(self.multiple_ood_datasets)
                figs[f"multiple_fig_roc_{self.ood_name}_ood"] = multiple_roc_fig
            else:
                fpr_at_recall_level, aupr_in, aupr_out, auroc, fig_roc = self.ood_evaluation()  # fig_aupr_in, fig_aupr_out
                
                summary_metrics[f"fpr_at_tpr_{self.ood_name}_ood"] = fpr_at_recall_level
                summary_metrics[f"aupr_in_{self.ood_name}_ood"] = aupr_in
                summary_metrics[f"aupr_out_{self.ood_name}_ood"] = aupr_out
                summary_metrics[f"auroc_{self.ood_name}_ood"] = auroc
                figs[f"fig_roc_{self.ood_name}_ood"] = fig_roc
                # figs[f"fig_aupr_in_{self.ood_name}_ood"] = fig_aupr_in
                # figs[f"fig_aupr_out_{self.ood_name}_ood"] = fig_aupr_out
            if self.edl_unc_exists:
                print("[INFO]: Calculating OOD with EDL Uncertainty")
                fpr_at_recall_level_unc, aupr_in_unc, aupr_out_unc, auroc_unc = self.ood_unc_evaluation()  # fig_roc_unc
                summary_metrics[f"fpr_at_tpr_{self.ood_name}_unc_ood"] = fpr_at_recall_level_unc
                summary_metrics[f"aupr_in_{self.ood_name}_unc_ood"] = aupr_in_unc
                summary_metrics[f"aupr_out_{self.ood_name}_unc_ood"] = aupr_out_unc
                summary_metrics[f"auroc_{self.ood_name}_unc_ood"] = auroc_unc   
                # figs[f"fig_roc_{self.ood_name}_unc_ood"] = fig_roc_unc 
        # Calculate Misclassifications
        if self.calc_misclassification:
            print("[INFO]: Calculating Misclassifications")
            misclassification_hist_fig = self.misclassification_evaluation()
            figs[f"fig_hist_{self.misclassification_name}_misclass"] = misclassification_hist_fig
           # figs[f"fig_roc_{self.misclassification_name}_misclass"] = fig_roc 
            # fpr_at_recall_level, aupr_in, aupr_out, auroc, fig_roc = self.misclassification_evaluation()  # fig_aupr_in, fig_aupr_out
            # summary_metrics[f"fpr_at_tpr_{self.misclassification_name}_misclass"] = fpr_at_recall_level
            # summary_metrics[f"aupr_in_{self.misclassification_name}_misclass"] = aupr_in
            # summary_metrics[f"aupr_out_{self.misclassification_name}_misclass"] = aupr_out
            # summary_metrics[f"auroc_{self.misclassification_name}_misclass"] = auroc
            # figs[f"fig_roc_{self.misclassification_name}_misclass"] = fig_roc
        # Plot Entropy distributions of OOD and ID pixels for baseline and improv models.
        if self.plot_id_ood_distributions:
            print("[INFO]: Plotting Distributions")
            ood_id_entropy_dist_fig = plot_entropy_distribution(self.ood_labels_dir, self.ood_entropy_baseline_dir, self.ood_entropy_improv_baseline_dir, baseline_mdl_title=self.baseline_mdl_title, improv_mdl_title=self.improv_mdl_title, title=f"Entropy Distribution of {self.baseline_mdl_title} vs {self.improv_mdl_title} Dataset.")
            figs["fig_ood_id_entropy_dist"] = ood_id_entropy_dist_fig
        # Calculate remaining Segmentation metrics
        if self.calc_seg_metrics:
            print("[INFO]: Calculating Seg Metrics")
            seg_metrics, fig_per_imgs = self.segmentation_evaluation()
            figs["per_img_metrics"] = fig_per_imgs
            # Plot the per class metrics (unc / entropy vs accuracy, unc / entropy vs iou, unc / entropy vs frequency)
            acc_vs_entropy_fig = plot_per_class_metrics(seg_metrics["per_category_accuracy"], seg_metrics["per_category_entropy"], metric1_title="Accuracy", metric2_title="Entropy", class_labels=self.labels)
            iou_vs_entropy_fig = plot_per_class_metrics(seg_metrics["per_category_iou"], seg_metrics["per_category_entropy"], metric1_title="IoU", metric2_title="Entropy", class_labels=self.labels)
            entropy_vs_freq_fig = plot_per_class_metrics(seg_metrics["per_category_entropy"], seg_metrics["per_category_freq"], metric1_title="Entropy", metric2_title="Frequency", class_labels=self.labels)
            if seg_metrics["per_category_mcd_unc"] is not None:
                acc_vs_MCDunc_fig = plot_per_class_metrics(seg_metrics["per_category_accuracy"], seg_metrics["per_category_mcd_unc"], metric1_title="Accuracy", metric2_title="MCD Prediction_Variance", class_labels=self.labels)
                iou_vs_MCDunc_fig = plot_per_class_metrics(seg_metrics["per_category_iou"], seg_metrics["per_category_mcd_unc"], metric1_title="IoU", metric2_title="MCD Prediction_Variance", class_labels=self.labels)
                MCDunc_vs_freq_fig = plot_per_class_metrics(seg_metrics["per_category_mcd_unc"], seg_metrics["per_category_freq"], metric1_title="MCD Prediction_Variance", metric2_title="Frequency", class_labels=self.labels)
                figs["fig_per_class_ACC_vs_MCDUnc"] = acc_vs_MCDunc_fig
                figs["fig_per_class_IoU_vs_MCDUnc"] = iou_vs_MCDunc_fig
                figs["fig_per_class_MCDUnc_vs_FREQ"] = MCDunc_vs_freq_fig
            # if seg_metrics["per_category_edl_belief"] is not None:
            #     acc_vs_EDLbelief_fig = plot_per_class_metrics(seg_metrics["per_category_accuracy"], seg_metrics["per_category_edl_belief"], metric1_title="Accuracy", metric2_title="EDL Belief", class_labels=self.labels)
            #     iou_vs_EDLbelief_fig = plot_per_class_metrics(seg_metrics["per_category_iou"], seg_metrics["per_category_edl_belief"], metric1_title="IoU", metric2_title="EDL Belief", class_labels=self.labels)
            #     EDLbelief_vs_freq_fig = plot_per_class_metrics(seg_metrics["per_category_edl_belief"], seg_metrics["per_category_freq"], metric1_title="EDL Belief", metric2_title="Frequency", class_labels=self.labels)
            #     figs["fig_per_class_ACC_vs_EDLBelief"] = acc_vs_EDLbelief_fig
            #     figs["fig_per_class_IOU_vs_EDLBelief"] = iou_vs_EDLbelief_fig
            #     figs["fig_per_class_EDLBelief_vs_FREQ"] = EDLbelief_vs_freq_fig
            if seg_metrics["per_category_edl_unc"] is not None:
                acc_vs_EDLUnc_fig = plot_per_class_metrics(seg_metrics["per_category_accuracy"], seg_metrics["per_category_edl_unc"], metric1_title="Accuracy", metric2_title="EDL Uncertainty", class_labels=self.labels)
                iou_vs_EDLUnc_fig = plot_per_class_metrics(seg_metrics["per_category_iou"], seg_metrics["per_category_edl_unc"], metric1_title="IoU", metric2_title="EDL Uncertainty", class_labels=self.labels)
                EDLUnc_vs_freq_fig = plot_per_class_metrics(seg_metrics["per_category_edl_unc"], seg_metrics["per_category_freq"], metric1_title="EDL Uncertainty", metric2_title="Frequency", class_labels=self.labels)
                figs["fig_per_class_ACC_vs_EDLUnc"] = acc_vs_EDLUnc_fig
                figs["fig_per_class_IOU_vs_EDLUnc"] = iou_vs_EDLUnc_fig
                figs["fig_per_class_EDLUnc_vs_FREQ"] = EDLUnc_vs_freq_fig

            figs["fig_per_class_acc_vs_entropy"] = acc_vs_entropy_fig
            figs["fig_per_class_iou_vs_entropy"] = iou_vs_entropy_fig
            figs["fig_per_class_entropy_vs_freq"] = entropy_vs_freq_fig
            # Add all the segmentation metrics to summary metrics for further logging to wandb
            summary_metrics.update(seg_metrics)
        print(f"[INFO]: Summary Metrics:\n {summary_metrics}")
        # Log Metrics to Wandb
        # FIXME: HANDLE logging for different metrics running indiviudaly to the same offline run
        self.run = log_eval_metrics(self.run, self.scalar_metrics_to_log, self.per_category_metrics_to_log, self.labels, summary_metrics, figs, self.plots_save_dir)
        return summary_metrics, figs, self.run

if __name__ == "__main__":
    # Load args and run configurations
    args = ...  # Load your arguments here
    run = ...  # Load your run configurations here
    rank = ... # Load yiu gpu id here

    test_metrics = TestMetrics(args=args, run=run, gpu_id=rank)
    summary_metrics, figs, run = test_metrics.main()
