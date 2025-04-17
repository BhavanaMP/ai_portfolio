import os
import random
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from evaluation.evaluation_utils import get_labels_data, load_model_for_evaluation, check_dirs_existence, load_test_dataset, save_data, load_image, load_mask, preprocess_image, pad_tensor
from modelling.model_utils import enable_dropout
from posthoc.posthoc_utils import load_temperature
from utils.plot_utils import plot_predictions, plot_predictions_individually


class Evaluation:
    def __init__(self, args, run, gpu_id):
        self.model_path = args.model_path
        self.model_names = args.model_names
        print(f"MODEL Names: {self.model_names}")
        self.dataset_name = args.eval_dataset_name
        self.model_type = args.eval_model_type

        # # Get id2labels
        # self.id2label, _, _ = get_labels_data(dataset_name=self.dataset_name)
        # we get the usual trained dataset labels
        if "cityscapes" in self.model_path:
            dataset_name = "cityscapes"
        elif "railsem19" in self.model_path:
            dataset_name = "railsem19"
        self.id2label, _, _ = get_labels_data(dataset_name=dataset_name)
        self.num_classes = len(self.id2label)
        # Original Trained Model
        ood_eval = True if args.run_ood_inference else False
        self.model = load_model_for_evaluation(self.model_path, self.model_names, self.num_classes, model_type=self.model_type, ood_eval=ood_eval)

        # Path to the test dataset and also robustness test path for railsem19
        self.run_robustness_inference = args.run_robustness_inference
        self.robustness_data_path = args.robustness_data_path_hf
        if self.run_robustness_inference and self.dataset_name != "railsem19":
            raise ValueError(f"Run robustness Inference is {self.run_robustness_inference} but given Invalid dataset name: {self.dataset_name}. Robustness Inference is only possible for railsem19 dataset.")

        # OOD Test Path
        self.run_ood_inference = args.run_ood_inference
        allowed_ood_datasets = ["fishyscapes", "lostandfound", "lostandfound_all", "obstacles", "anomaly", "streethazards"]
        if self.run_ood_inference and self.dataset_name.lower() not in allowed_ood_datasets:
            raise ValueError(f"Run OOD Inference is {self.run_ood_inference} but given Invalid dataset name: {self.dataset_name}. OOD Inference is possible for {allowed_ood_datasets}")

        # MCD Inference
        self.run_mcdropout_inference = args.run_mcdropout_inference

        # Get the dataset
        self.test_dataset = load_test_dataset(self.dataset_name)

        # We make the test set batch size to 1 by default
        self.batch_size = 1
        self.is_distributed = args.is_distributed
        self.device = gpu_id if self.is_distributed else "cpu"

        # Temperature scaling
        self.use_tempsscaling = args.use_tempsscaling
        if self.use_tempsscaling:
            self.temperature = load_temperature(args.saved_temperature_path)
            print(f"[INFO]: Using temperature scaling on test set to calibrate.Temperature: {self.temperature}")

        # Directories Management
        file_name = os.path.basename(self.model_path)
        # Get the model name from the checkpoint path to construct the save directory
        self.save_folder_name = os.path.splitext(file_name)[0][6:]
        self.base_dir = f"TestResults/{self.save_folder_name}/{self.model_type}"
        print(self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        # Customize the subdirs
        if self.dataset_name == "railsem19" and self.run_robustness_inference:
            # Get the robustness test name. Robustness Inference only exist for Railsem19
            test_name = self.robustness_data_path.split('split_')[1]
            self.output_dir = f"{self.base_dir}/Robustness_{test_name}"
        # OOD Data Test Inference
        elif self.run_ood_inference:
            # Get the OOD dataset name
            ood_name = self.dataset_name
            self.output_dir = f"{self.base_dir}/OOD_{ood_name}"
            if self.run_mcdropout_inference and self.model_type != "edl":
                # MCD for OOD Data Test Inference. We only run for MCD Trained Model. EDL doesnt support MCDInference
                self.output_dir = f"{self.base_dir}/OOD_{ood_name}/MCDInference"
                # Only gets for MCD inference
                self.forward_passes = 10
                self.pred_var_output_dir = f"{self.output_dir}/pred_var"
                check_dirs_existence([self.pred_var_output_dir])
            else:
                self.output_dir = f"{self.base_dir}/OOD_{ood_name}"
        # MCD for original In distribution Test inference
        elif self.run_mcdropout_inference and self.model_type != "edl":
            self.output_dir = f"{self.base_dir}/MCDInference"
            # Only gets for MCD inference
            self.forward_passes = 10
            self.pred_var_output_dir = f"{self.output_dir}/pred_var"
            check_dirs_existence([self.pred_var_output_dir])
        else:
            # Original In distribution Test Inference
            self.output_dir = f"{self.base_dir}/Original"

        if self.model_type == "edl":
            self.edl_uncertainty_output_dir = f"{self.output_dir}/edl_uncertainty"
            # self.edl_beliefs_output_dir = f"{self.output_dir}/edl_belief"
            check_dirs_existence([self.edl_uncertainty_output_dir])  # self.edl_beliefs_output_dir

        print(f"Test Output Directory: {self.output_dir}")

        # Final Save directories paths
        self.labels_output_dir = f"{self.output_dir}/labels"
        self.probs_output_dir = f"{self.output_dir}/probs"
        self.entropy_output_dir = f"{self.output_dir}/entropies"
        self.preds_output_dir = f"{self.output_dir}/preds"
        self.max_pred_probs_output_dir = f"{self.output_dir}/max_pred_probs"

        # Make sure the save directories exists
        check_dirs_existence(dirs_to_create=[self.labels_output_dir, self.probs_output_dir, self.entropy_output_dir, self.preds_output_dir, self.max_pred_probs_output_dir])

        # Log to wandb for test evaluations.
        if run is not None:
            run.config["classes"] = len(self.id2label)
            self.run = run

        # For prediction
        if args.run_mcd_for_preds:
            self.forward_passes = 10

        # Move model to GPU
        self.model.to(self.device)
        # Wrapping model into DDP
        if self.is_distributed:
            print("[INFO]: Wrapping the model into DDP")
            self.model = DDP(module=self.model, device_ids=[gpu_id])

    def setup_mcdropout(self):
        """
        Function to set up MCDropout directories and model configuration
        """
        # Enable dropout layers in inference mode
        model = enable_dropout(self.model)
        return model

    def calc_entropy(self, probs, dim=1):
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=dim)
        return entropy / torch.log(torch.tensor(probs.shape[dim], dtype=torch.float32))  # 1 / log K Σ k=1toN -pk_i log (pk_i)

    @torch.no_grad()
    def calculate_predictive_uncertainty(self, images):
        print(f"[INFO]: Performing forward pass of MC Dropout... {self.model.training}, {torch.is_grad_enabled()}")
        num_classes = len(self.id2label)
        mc_logits = torch.empty(size=(self.forward_passes, len(images), num_classes, images.size(-2), images.size(-1)), device=self.device)
        # Enable MC Dropout
        model = self.setup_mcdropout()
        for i in range(self.forward_passes):
            mylogits = model(images)
            mc_logits[i] = mylogits
            del mylogits
        if self.use_tempsscaling:
            mc_logits = mc_logits / self.temperature
        y_prob_samples = torch.softmax(mc_logits, dim=2)
        preds_probs = torch.mean(y_prob_samples, dim=0)
        # Usual std predictive_uncertainty
        preds_var = torch.std(torch.softmax(mc_logits, dim=2), dim=0)
        max_probs, preds = torch.max(preds_probs, dim=1)
        predictive_uncertainty = torch.empty((images.shape[0], images.shape[2], images.shape[3]), device=self.device)
        for i in range(num_classes):
            predictive_uncertainty = torch.where(preds == i, preds_var[:, i, :, :], predictive_uncertainty)
        predictive_entropy = self.calc_entropy(preds_probs, dim=1)
        return preds_probs, max_probs, preds, predictive_uncertainty, predictive_entropy

    def test_step(self, test_dataloader):
        self.model.eval()
        for _, batch in enumerate(tqdm(test_dataloader)):
            images = batch["image"].to(self.device)
            gt_masks = batch["mask"].to(self.device)
            image_name = batch["name"][0]  # list[names]
            # Usual Inference
            with torch.no_grad():
                if self.model_type in ["mcd", "baseline"]:
                    if self.run_mcdropout_inference:
                        # RUN MCDropout inference during testing to get epistemic uncertainties
                        preds_probs, max_probs, preds, predictive_uncertainty, predictive_entropy = self.calculate_predictive_uncertainty(images)
                        save_data(self.pred_var_output_dir, predictive_uncertainty, image_name)
                    else:
                        test_logits = self.model(images)
                        if self.use_tempsscaling:
                            # Use Temp scaling
                            test_logits = test_logits / self.temperature
                        preds_probs = F.softmax(test_logits, dim=1)
                        max_probs, preds = torch.max(preds_probs, dim=1)
                        predictive_entropy = self.calc_entropy(preds_probs, dim=1)
                elif self.model_type == "edl":
                    # FIXME: Handle Predictive Uncertainty here
                    if self.is_distributed:
                        preds_probs, uncertainty, beliefs = self.model.module.predict(images, return_uncertainty=True)
                    else:
                        preds_probs, uncertainty, beliefs = self.model.predict(images, return_uncertainty=True)
                    max_probs, preds = torch.max(preds_probs, dim=1)
                    predictive_entropy = self.calc_entropy(preds_probs, dim=1)
                    # save_data(self.edl_uncertainty_output_dir, uncertainty, image_name)
                    # ####save_data(self.edl_beliefs_output_dir, beliefs, image_name)
                else:
                    raise ValueError(f"model_type should be any of ['mcd', 'edl', 'baseline'], but given {self.model_type}")

            # Save probs
            save_data(self.probs_output_dir, preds_probs, image_name)
            # # Save the Maximum prediction probabilities
            # save_data(self.max_pred_probs_output_dir, max_probs, image_name)
            # # Save the preds
            # save_data(self.preds_output_dir, preds, image_name)
            # # Save the entropy
            # save_data(self.entropy_output_dir, predictive_entropy, image_name)
            # # Save the true labels
            # save_data(self.labels_output_dir, gt_masks, image_name)
            del batch

    def test_eval(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)
        self.test_step(test_dataloader)
        print(f"\n[INFO]: Test Evaluation Done!!!!")

    # Prediction
    def predict(self, inputs: Tuple[str, str] = None, choose_random: bool = False, plot_individual: bool=False, selected_test_image:  Union[str, None] = None, run_mcd_inference: bool = False, model_type: str="mcd", fig_title: str = "prediction"):
        print("[INFO]: Predicting the image...")
        # if not any([choose_random, inputs is not None, selected_test_image is not None]):
        if not any([choose_random, inputs, selected_test_image]):
            raise ValueError("Either inputs or choose_random or selected_test_image must be provided.")
        
        print(inputs, type(inputs), choose_random, selected_test_image, type(selected_test_image))

        if choose_random:
            # Choose random index from test dataset
            idx = random.randint(0, len(self.test_dataset) - 1)
            image = self.test_dataset[idx]["image"]
            gt_mask = self.test_dataset[idx].get("mask", None)
            original_height = self.test_dataset[idx]["original_height"]
            original_width = self.test_dataset[idx]["original_width"]
            image_name = self.test_dataset[idx].get("name")
            fig_title = f"prediction_{image_name}"
            inputs = [(image, gt_mask)]
        elif selected_test_image is not None: 
            # Take image name of the test set from the user if given and predict
            print(f"selected_test_image: {selected_test_image}")
            selected_input = None
            for idx in range(len(self.test_dataset)):
                data = self.test_dataset[idx]
                if data["name"].lower() == selected_test_image.lower() or selected_test_image.lower() in data["name"].lower():
                    selected_input = self.test_dataset[idx]
                    inputs = [(selected_input["image"], selected_input["mask"])]
                    original_height = self.test_dataset[idx]["original_height"]
                    original_width = self.test_dataset[idx]["original_width"]
                    fig_title = f"prediction_{selected_test_image}"
                    break
            if selected_input is None:
                raise ValueError("Provided image name not found in the dataset.")
        elif inputs is not None:
            # take images from the path
            image_path = inputs[0]
            gt_mask_path = inputs[1]
            if isinstance(image_path, str):  # If its path
                image = load_image(image_path)
                original_width, original_height = image.size
                image_name = os.path.splitext(os.path.basename(image_path))[0]
            gt_mask = None
            if isinstance(gt_mask_path, str):  # If the gt_mask path exists
                gt_mask = load_mask(gt_mask_path)
            # Preprocess here when loading from given path
            image_tensor, mask_tensor = preprocess_image(self.dataset_name, image, gt_mask)
            inputs = [(image_tensor, mask_tensor)]

        for image_tensor, mask_tensor in inputs:
            predictive_uncertainty, edl_uncertainty, edl_belief = None, None, None
            # Padding
            if self.dataset_name == "railsem19":
                # For all other datasets, we already does padding in the transforms
                image_tensor, _ = pad_tensor(image_tensor, value=0)
                if mask_tensor is not None:
                    mask_tensor, _ = pad_tensor(mask_tensor, value=255)  # Assuming 255 is the ignore index for masks

            image_tensor = image_tensor.to(self.device)
            image_tensor = image_tensor.unsqueeze(dim=0)
            if mask_tensor is not None:
                mask_tensor = mask_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
                mask_tensor = mask_tensor.to(self.device)
                print(f"gt_mask: {mask_tensor.shape}")  # [1, 1, h, w])

            print(f"image: {image_tensor.shape}")  # [1, 3, h, w], 

            with torch.no_grad():
                if model_type.lower() in ["mcd", "baseline"]:
                    if run_mcd_inference is not None and run_mcd_inference:
                        preds_probs, max_probs, preds, predictive_uncertainty, predictive_entropy = self.calculate_predictive_uncertainty(image_tensor)
                        print(f"predictive_uncertainty: {predictive_uncertainty.shape}")
                    else:
                        logits = self.model(image_tensor)
                        # FIXME: Use temp scaling for predictions too?
                        preds_probs = F.softmax(logits, dim=1)
                        max_probs, preds = torch.max(preds_probs, dim=1)
                        predictive_entropy = self.calc_entropy(preds_probs, dim=1)
                        print(f"logits: {logits.shape}")
                elif model_type.lower() == "edl":
                    if self.is_distributed:
                        preds_probs, edl_uncertainty, edl_belief = self.model.module.predict(image_tensor, return_uncertainty=True)
                    else:
                        preds_probs, edl_uncertainty, edl_belief = self.model.predict(image_tensor, return_uncertainty=True)
                    max_probs, preds = torch.max(preds_probs, dim=1)
                    predictive_entropy = self.calc_entropy(preds_probs, dim=1)
                    edl_uncertainty = edl_uncertainty.squeeze(dim=1)
                    print(f"edl_uncertainty: {edl_uncertainty.shape}, edl_belief: {edl_belief.shape}")
                else:
                    raise ValueError(f"model_type should be any of ['mcd', 'edl', 'baseline'], but given {model_type}")    
            
            print(f"preds_probs: {preds_probs.shape}, max_probs: {max_probs.shape}, preds: {preds.shape}, predictive_entropy: {predictive_entropy.shape}")
            
            pred_mask = preds.squeeze(dim=0).cpu().numpy()
            max_probs_map = max_probs.squeeze(dim=0).cpu().numpy()
            entropy_map = predictive_entropy.squeeze(dim=0).cpu().numpy()

            if mask_tensor is not None:
                gt_mask_array = mask_tensor.squeeze().cpu().numpy()
                binary_error_map = (preds.squeeze(dim=0).cpu().numpy() != gt_mask_array).astype(np.uint8)
            else:
                gt_mask_array = None
                binary_error_map = None

            uncertainty_map = None
            belief = False
            
            if predictive_uncertainty is not None:
                uncertainty_map = predictive_uncertainty.squeeze(dim=0).cpu().numpy()
            if edl_uncertainty is not None:
                uncertainty_map = edl_uncertainty.squeeze(dim=0).cpu().numpy()
                # belief = edl_belief.squeeze(dim=0).cpu().numpy()
                belief = True

            # Plot the prediction
            if not plot_individual:
                figs = plot_predictions(image_tensor.cpu(), gt_mask_array, pred_mask, binary_error_map, max_probs_map,
                                        entropy_map, uncertainty_map, belief, original_height, original_width, self.dataset_name)
                # Save the figure as a PNG file
                figs.savefig(f"./prediction_plots/{self.save_folder_name}_{self.dataset_name}_{fig_title}.pdf", format="pdf")
                # Log to wandb
                self.run.log({f"{fig_title}": figs})
                print(f"Single prediction figure saved in the path ./prediction_plots/{self.save_folder_name}_{self.dataset_name}_{fig_title}.png")
            else:
                figs = plot_predictions_individually(image_tensor.cpu(), gt_mask_array, pred_mask, binary_error_map,
                                                     max_probs_map, entropy_map, uncertainty_map, belief, 
                                                     original_height, original_width, self.dataset_name)
                base_dir = f"./prediction_plots/{self.save_folder_name}_{self.dataset_name}"
                output_dir = os.path.join(base_dir, fig_title)
                # Ensure directories exist
                os.makedirs(output_dir, exist_ok=True)
                print(f"Saving the prediction plots at {output_dir}")
                for name, fig in figs.items():
                    try:
                        file_path = os.path.join(output_dir, f"{name}.pdf") 
                        fig.savefig(file_path, bbox_inches='tight', format="pdf") # .png
                        # # Close the figure after saving to free up memory
                        # plt.close(fig)  
                        print(f"Saved: {file_path}")
                    except Exception as e:
                        print(f"Error saving {name}: {e}")
                print(f"All figures have been saved to {output_dir}")
