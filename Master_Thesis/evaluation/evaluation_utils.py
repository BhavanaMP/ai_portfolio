import os

import torch
import torch.nn.functional as F
import wandb
import numpy as np
import pandas as pd
from PIL import Image
import plotly.io as pio
pio.kaleido.scope.mathjax = None
from plotly.io import write_image

from data.prepare_anamoly import get_anamoly_labels, load_anamoly_test_dataset, CustomAnamolyDataset, get_anamoly_test_transforms
from data.prepare_cityscapes_hf import get_cityscapes_labels, load_cityscapes, CustomCityscapesDataset, get_cityscapes_test_transforms
from data.prepare_fishyscapes import get_fishyscapes_labels, load_fishyscapes, CustomFishyscapesDataset, get_fishyscapes_test_transforms
from data.prepare_lostandfound import get_lostandfound_labels, load_lostandfound, CustomLostAndFoundDataset, get_lostandfound_test_transforms
from data.prepare_lostandfound_full import get_lostandfound_full_labels, load_lostandfound_full, CustomLostAndFoundFullDataset, get_lostandfound_full_test_transforms
from data.prepare_obstacles import get_obstacles_labels, load_obstacles, CustomObstaclesDataset, get_obstacles_test_transforms
from data.prepare_railsem19 import get_railsem19_labels, load_railsem19_test_dataset, CustomRailsem19Dataset, get_railsem19_transforms
from data.prepare_streethazards import get_streethazards_labels, load_streethazards, CustomStreetHazardsDataset, get_streethazards_test_transforms

from modelling.edl_classifier import EDLClassifier
from modelling.mcd_classifier import MCDClassifier


def load_test_dataset(dataset_name):
    """
    Load Test dataset for evaluation
    """
    if dataset_name == "railsem19":
        print(f"[INFO]: Loading {dataset_name} test dataset")
        test_dataset = load_railsem19_test_dataset()["test"]
        transforms = get_railsem19_transforms()
        test_dataset = CustomRailsem19Dataset(test_dataset, transforms, split="test")
    elif dataset_name == "cityscapes":
        print(f"[INFO]: Loading {dataset_name} test dataset")
        # Note: For cityscapes its val, test GT is not publicly available
        test_dataset = load_cityscapes(split="validation")
        # test_dataset = test_dataset.select(indices=range(4))
        # We apply Test transforms though
        test_dataset = CustomCityscapesDataset(test_dataset, split="test")
    elif dataset_name == "fishyscapes":
        print(f"[INFO]: Loading {dataset_name} dataset")
        test_dataset = load_fishyscapes()
        test_dataset = CustomFishyscapesDataset(test_dataset)
    elif dataset_name == "lostandfound":
        print(f"[INFO]: Loading {dataset_name} dataset")
        test_dataset = load_lostandfound()
        test_dataset = CustomLostAndFoundDataset(test_dataset)
    elif dataset_name == "lostandfound_full":
        print(f"[INFO]: Loading {dataset_name} dataset")
        test_dataset = load_lostandfound_full(split="test")
        test_dataset = CustomLostAndFoundFullDataset(test_dataset, split="test")
    elif dataset_name == "obstacles":
        print(f"[INFO]: Loading {dataset_name} dataset")
        test_dataset = load_obstacles()
        test_dataset = CustomObstaclesDataset(test_dataset)
    elif dataset_name == "anomaly":
        print(f"[INFO]: Loading {dataset_name} dataset")
        test_dataset = load_anamoly_test_dataset()
        test_dataset = CustomAnamolyDataset(test_dataset)
    elif dataset_name == "streethazards":
        print(f"[INFO]: Loading {dataset_name} dataset")
        test_dataset = load_streethazards()
        test_dataset = CustomStreetHazardsDataset(test_dataset)
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name} given for evaluation. Allowed are ["railsem19", "cityscapes", "fishyscapes", "lostandfound", "lostandfound_full", "obstacles", "anomaly", "streethazards"]')
    print(f"[INFO]: Total Test images: {len(test_dataset)}")
    return test_dataset


def load_model_for_evaluation(model_path, model_names, num_classes, model_type, ood_eval=False):
    """
    Get the Model from training checkpoint for evaluation.
    """
    print("[INFO]: Getting Model for Test Evaluation.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint Path doesn't exist: {model_path}")
    loaded_ckpt = torch.load(f=model_path)
    encoder, decoder = model_names
    if model_type.lower() == "edl":
        # Get the EDL classifier
        print("[INFO]: Getting the EDL Classifer")
        model = EDLClassifier(num_classes=num_classes, encoder=encoder, decoder=decoder)
    elif model_type.lower() in ["mcd", "baseline"]:
        # Get the MCD classifier
        print("[INFO]: Getting the MCD Classifer")
        model = MCDClassifier(num_classes=num_classes, encoder=encoder, decoder=decoder)
    if set(loaded_ckpt["model_state_dict"].keys()) != set(model.state_dict().keys()):
        raise KeyError("The loaded checkpoint does not contain the expected keys.")
    model.load_state_dict(loaded_ckpt["model_state_dict"], strict=not ood_eval)
    return model


def get_preprocess_transfroms(dataset_name):
    if dataset_name == "railsem19":
        image_transforms, mask_transforms = get_railsem19_transforms()
        return image_transforms, mask_transforms
    elif dataset_name == "cityscapes":
        transforms = get_cityscapes_test_transforms()
    elif dataset_name == "lostandfound":
        transforms = get_lostandfound_test_transforms()
    elif dataset_name == "lostandfound_full":
        transforms = get_lostandfound_full_test_transforms()
    elif dataset_name == "fishyscapes":
        transforms = get_fishyscapes_test_transforms()
    elif dataset_name == "obstacles":
        transforms = get_obstacles_test_transforms()
    elif dataset_name == "anamoly":
        transforms = get_anamoly_test_transforms()
    elif dataset_name == "streethazards":
        transforms = get_streethazards_test_transforms()
    else:
        raise ValueError(f"Unable to retrieve transforms. Invalid dataset name: {dataset_name}")
    return transforms


def save_data(save_dir, tensor_data, image_name):
    """
    Function to save the test metrics as npz files
    """
    # numpy
    save_path = os.path.join(save_dir, f"{image_name}.npz")
    np.savez_compressed(save_path, data=tensor_data.cpu().detach().numpy())


def load_prediction(save_dir, image_name):
    """
    Function to load the saved data
    """
    save_path = os.path.join(save_dir, f"{image_name}.npz")
    data = np.load(save_path)
    prediction = data["data"]
    prediction = torch.from_numpy(prediction)
    return prediction


def check_dirs_existence(dirs_to_create: list):
    """
    Function that checks if the directories to save the test npz files exists.

    If not, creates the missing directories.
    """
    # Create directories if they do not exist
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' created or already exists.")


def get_labels_data(dataset_name="railsem19"):
    """
    Get the labels of the datasets
    """
    # Get id2label, label2id, labels
    if dataset_name == "railsem19":
        id2label, label2id, labels = get_railsem19_labels()
    elif dataset_name == "cityscapes":
        id2label, label2id, labels = get_cityscapes_labels()
    elif dataset_name == "lostandfound":
        id2label, label2id, labels = get_lostandfound_labels()
    elif dataset_name == "lostandfound_full":
        id2label, label2id, labels = get_lostandfound_full_labels()
    elif dataset_name == "fishyscapes":
        id2label, label2id, labels = get_fishyscapes_labels()
    elif dataset_name == "obstacles":
        id2label, label2id, labels = get_obstacles_labels()
    elif dataset_name == "anamoly":
        id2label, label2id, labels = get_anamoly_labels()
    elif dataset_name == "streethazards":
        id2label, label2id, labels = get_streethazards_labels()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return id2label, label2id, labels


def load_npz_files(dir_path, convert_to_torch=True):
    """Helper function to load and concatenate data from all saved npz files in a directory."""
    # Ensure directory exists
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory '{dir_path}' does not exist.")

    all_data = []
    for file_name in sorted(os.listdir(dir_path)):
        if file_name.endswith(".npz"):
            file_path = os.path.join(dir_path, file_name)
            data = np.load(file_path)["data"]
            all_data.append(data)

    # Ensure there is data to concatenate
    if len(all_data) == 0:
        raise ValueError(f"No .npz files found in directory '{dir_path}'.")

    # Concatenate and convert data
    if convert_to_torch:
        concatenated_data = torch.from_numpy(np.concatenate(all_data, axis=0))
    else:
        concatenated_data = np.concatenate(all_data, axis=0)

    return concatenated_data


def save_per_img_results(imgs_IoUs, imgs_accs, imgs_ece, imgs_nlls, imgs_names, save_file_name):
    """
    Saves the Test set Per Image results tp later use them for statistical significance test.
    - Per Image IoU
    - Per Image Accuracy
    - Per Image NLL
    - Per Image Expected Calibration Error
    - Per Image Name
    """
    try:
        print(f"[INFO]: Trying to save test results at: {save_file_name}")
        
        # Create a dictionary with mandatory columns
        data = {
            "test_img_names": imgs_names,
            "test_IoUs": imgs_IoUs,
            "test_accs": imgs_accs,
            "test_nlls": imgs_nlls
        }
        
        # Include imgs_ece only if it exists
        if imgs_ece:
            data["test_eces"] = imgs_ece
        
        # Create DataFrame and save to CSV
        test_results_df = pd.DataFrame(data)
        test_results_df.to_csv(save_file_name, index=False)
        
        print("[INFO]: Test results saved successfully.")
    except Exception as e:
        print(f"[WARN]: Failed to save per image test results: {e}")


# Prediction Related Functions
def pad_tensor(image, mode='constant', value=255):
    if (image.shape[1] % 32 != 0 or image.shape[2] % 32 != 0):
        height, width = image.shape[1], image.shape[2]
        # Calculate padding to ensure both height and width are divisible by 32
        pad_height = max(32 - height % 32, 0)
        pad_width = max(32 - width % 32, 0)
        pad_height_half = pad_height // 2
        pad_width_half = pad_width // 2
        pad = (pad_width_half, pad_width - pad_width_half, pad_height_half, pad_height - pad_height_half)
    # Pad the images
    padded_image = F.pad(image, pad=pad, mode=mode, value=value)
    return padded_image, pad


def unpad_tensor(padded_image, pad):
    _, _, height, width = padded_image.shape
    pad_width_half, pad_width, pad_height_half, pad_height = pad
    # Calculate the original image dimensions
    original_height = height - pad_height - pad_height_half
    original_width = width - pad_width - pad_width_half
    # Unpad the image to its original dimensions
    unpadded_image = padded_image[:, :, pad_height_half:pad_height_half + original_height, pad_width_half:pad_width_half + original_width]
    return unpadded_image


def load_image(fpath):
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Image path {fpath} does not exist")
    return Image.open(fpath).convert('RGB')


def load_mask(path):
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask path {path} does not exist")
        return Image.open(path).convert('L')
    return None


def preprocess_image(dataset_name, image, mask=None):
    """
    Preprocess the Image using transforms based on dataset_name of the trained model. This also works for unseen
    images where there is no ground truth.
    """
    if dataset_name == "railsem19":
        image_transforms, mask_transforms = get_preprocess_transfroms(dataset_name)
        # Add batch dimension
        transformed_image = image_transforms(image).unsqueeze(0)
        if mask is not None:
            # Convert to tensor and add batch and channel dimensions
            transformed_mask = mask_transforms(mask).unsqueeze(0).unsqueeze(0)
            return transformed_image, transformed_mask
        return transformed_image, None
    else:
        # All other datasets transforms are done using Albumentations which uses numpy arrays
        transforms = get_preprocess_transfroms(dataset_name)
        mask = np.array(mask) if mask is not None else None
        image = np.array(image)
        # print(f"image: {image.shape}, mask:{mask}")   # (1080, 1920, 3) None
        if mask is None:
            transformed_image = transforms(image=image)
            # print(transformed_image.keys(), transformed_image["image"].shape)  # dict_keys(['image']), torch.Size([3, 1088, 1920])
            # Add batch dimension
            transformed_image = transformed_image["image"]  # .unsqueeze(0)
            return transformed_image, None
        else:
            transformed_data = transforms(image=image, mask=mask)
            # Add batch dimension
            transformed_image = transformed_data["image"]  #.unsqueeze(0)
            # Add batch and channel dimensions
            transformed_mask = transformed_data["mask"]    #.unsqueeze(0).unsqueeze(0)
            return transformed_image, transformed_mask


def get_test_image_paths(folder_path):
    """
    This function is used to load the images from the local folder path for prediction.
    
    Make sure to place the jpg files in the folder and give that path
    """
    image_extensions = ['.jpg', '.jpeg', '.png',]
    image_paths = []
    for file_name in os.listdir(folder_path):
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, file_name))
    return image_paths


class LoadOutputsDataset(torch.utils.data.Dataset):
    """Load all outputs from a directory based on index."""
    def __init__(self, base_dir, pred_var_exists=False, edl_belief_exists=False):
        """
        Args:
            base_dir (str): The main directory where all outputs are stored.
                Examples:
                    "path_named_by_trainedmodel/Original"
                    "path_named_by_trainedmodel/MCDInference"
                    "path_named_by_trainedmodel/OOD_{ood_datasetname}"
                    "path_named_by_trainedmodel/OOD_{ood_datasetname}/MCDInference"
                    "path_named_by_trainedmodel/Robustness_{test_name}" # rain, fog, etc. This exists only for railsem19

            pred_var_exists (bool, optional): Indicates whether pred_var file exists (typically for MCD runs).
            pred_var_exists (bool, optional): Indicates whether edl_belief file exists (typically for EDL Models).
                Examples:
                    "path_named_by_trainedmodel/MCDInference"
                    "path_named_by_trainedmodel/OOD_{ood_datasetname}/MCDInference"
        """
        self.base_dir = base_dir
        self.pred_var_exists = pred_var_exists
        self.edl_belief_exists = edl_belief_exists

        self.subdirs = {
            "labels": os.path.join(base_dir, "labels"),
            "probs": os.path.join(base_dir, "probs"),
            "entropies": os.path.join(base_dir, "entropies"),
            "preds": os.path.join(base_dir, "preds"),
            "max_pred_probs": os.path.join(base_dir, "max_pred_probs"),
        }

        if pred_var_exists:
            self.subdirs["pred_var"] = os.path.join(base_dir, "pred_var")
        if edl_belief_exists:
            # self.subdirs["edl_belief"] = os.path.join(base_dir, "edl_belief")
            self.subdirs["edl_uncertainty"] = os.path.join(base_dir, "edl_uncertainty")

        # Check if directories exist
        for name, subdir in self.subdirs.items():
            if name != "edl_belief":
                if not os.path.isdir(subdir):
                    raise ValueError(f"Directory '{subdir}' does not exist.")

        # All directories have the same number of files with name as its image_name
        self.file_names = sorted(os.listdir(self.subdirs["labels"]))
        self.num_samples = len(self.file_names)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_dict = {}

        for name, subdir in self.subdirs.items():
            file_path = os.path.join(subdir, self.file_names[idx])
            npz_data = np.load(file_path)["data"]
            img_dict[name] = torch.from_numpy(npz_data)

        # Extract the base name without extension to get the image name
        img_dict["image_name"] = os.path.splitext(self.file_names[idx])[0]

        return img_dict


class LoadTwoOutputDataset(torch.utils.data.Dataset):
    """Load any 2 outputs from a directory based on index. Mainly useful to calculate ece and aurrc"""
    def __init__(self, base_dir, dir1_name="probs", dir2_name="labels"):
        """
        Args:
            base_dir (str): The main directory where all outputs are stored.
            dir1_name(str): The name of the first directory.
            dir2_name(str): The name of the second directory.
                Examples of base_dir:
                    "path_named_by_trainedmodel/Original"
                    "path_named_by_trainedmodel/MCDInference"
                    "path_named_by_trainedmodel/OOD_{ood_datasetname}"
                    "path_named_by_trainedmodel/OOD_{ood_datasetname}/MCDInference"
                    "path_named_by_trainedmodel/Robustness_{test_name}" # rain, fog, etc. This exists only for railsem19

                Examples of dir1_name and dir2_name:
                    "probs"
                    "labels"
        """
        self.base_dir = base_dir
        self.dir1_name = dir1_name
        self.dir2_name = dir2_name

        self.subdirs = {
            f"{self.dir1_name}": os.path.join(base_dir, f"{self.dir1_name}"),
            f"{self.dir2_name}": os.path.join(base_dir, f"{self.dir2_name}"),
        }

        # Check if directories exist
        for name, subdir in self.subdirs.items():
            if not os.path.isdir(subdir):
                raise ValueError(f"Directory '{subdir}' does not exist.")

        # Make sure to check all directories have the same number of files. This means this wont work different datsets realted outputs as test images differs for each dataset
        self.file_names_1 = sorted(os.listdir(self.subdirs[f"{self.dir1_name}"]))
        self.file_names_2 = sorted(os.listdir(self.subdirs[f"{self.dir2_name}"]))

        assert len(self.file_names_1) == len(self.file_names_2), "All directories in the LoadTwoOutputDataset must have the same number of files."

        self.num_samples = len(self.file_names_1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_dict = {}
        
        # print("getitem")
        # print(self.subdirs)  # {'probs': './TestResults/FPN_MIT_B3_baseline_cityscapes_t7z5h2i/baseline/Original/probs', 'labels': './TestResults/FPN_MIT_B3_baseline_cityscapes_t7z5h2i/baseline/Original/labels'}

        for idx, (name, subdir) in enumerate(self.subdirs.items()):
            file_path = os.path.join(subdir, self.file_names_1[idx])  # Assuming the file names are same
            npz_data = np.load(file_path)["data"]
            img_dict[name] = torch.from_numpy(npz_data)

        return img_dict


class LoadOODIDDataset(torch.utils.data.Dataset):
    """We need to load full dataset for OOD related metrics

    Args:
        ood_conf_file (str): Path to the npz file where each npzfile has OOD image confidence scores.
        ood_label_file (str): Path to the npz file where each npzfile has OOD image GT labels.
        
    """
    def __init__(self, ood_conf_file, ood_label_file):

        # Load all npz files from the ood_conf_file directory
        self.ood_conf_scores = load_npz_files(ood_conf_file)
        # Load all npz files from the ood_label_file directory. (0 for ID, 1 for OOD)
        self.ood_labels = load_npz_files(ood_label_file)

    def __len__(self):
        return len(self.ood_conf_scores)

    def __getitem__(self, idx):
        conf_score = self.ood_conf_scores[idx]
        target = self.ood_labels[idx]
        return {"scores": conf_score, "label": target}


class OODIDProbsDataset(torch.utils.data.Dataset):
    """We need to load full dataset for OOD & ID related metrics
    
    This class can be used when you want the OOD dataset images labelled as 1 as whole and another 
    In distribution dataset images labelled as 0 as whole.
    
    We are not using this in our metrics.

    Args:
            id_file (str): Path to the npz files directory where each npzfile has ID image confidence scores.
            ood_file (str): Path to the npz file where each npzfile has OOD image confidence scores.
    """
    def __init__(self, id_file, ood_file):

        # Load all npz files from the ID directory
        self.id_conf_scores = load_npz_files(id_file)
        # Load all npz files from the OOD directory
        self.ood_conf_scores = load_npz_files(ood_file)

        # Concatenate the ID and OOD scores
        self.conf_scores = torch.cat((self.id_conf_scores, self.ood_conf_scores), dim=0)

        # Create targets (0 for ID, 1 for OOD)
        self.targets = torch.cat((torch.zeros(len(self.id_conf_scores)), torch.ones(len(self.ood_conf_scores))), dim=0)

    def __len__(self):
        return len(self.conf_scores)

    def __getitem__(self, idx):
        conf_score = self.conf_scores[idx]
        target = self.targets[idx]
        return {"scores": conf_score, "label": target}
    

# class LoadMisclassificationDataset(torch.utils.data.Dataset):
#     """
#     Args:
#             conf_file (str): Path to the npz files directory of max_pred_probs.
#             pred_file (str): Path to the npz file directory of preds.
#             label_file (str): Path to the npz file directory of labels.
#     """
#     def __init__(self, conf_file, pred_file, label_file, is_ood):
#         print("Initializing Misclassification Dataset...")
#         # Load all npz files from the ID directory
#         self.conf_scores = load_npz_files(conf_file)  # convert_to_torch=False
#         # self.conf_scores = self.conf_scores.cpu()
        
#         if is_ood:
#             print("[INFO]: Getting the preds for OOD dataset")
#             self.preds = (self.conf_scores > 0.7).int()   # .astype(int)
#         else:
#             self.preds = load_npz_files(pred_file)   # , convert_to_torch=False
#         # self.preds = self.preds.cpu()
#         self.labels = load_npz_files(label_file)     # , convert_to_torch=False
#         # self.labels = self.labels.cpu()

#         # Create a boolean mask where True indicates a misclassification
#         misclassification_mask = self.preds != self.labels  # ).astype(int)
        
#         # Initialize targets with zeros, then set to 1 where misclassification occurs
#         # self.targets = torch.zeros_like(self.preds)   # , device="cpu"
#         # self.targets[misclassification_mask] = 1
#         # self.targets = self.targets.int()             # .cpu()
#         self.targets = misclassification_mask.int()
#         print(torch.unique(self.targets))

#     def __len__(self):
#         return len(self.conf_scores)

#     def __getitem__(self, idx):
#         conf_score = self.conf_scores[idx]
#         target = self.targets[idx]
#         print(conf_score, target)
#         return {"scores": conf_score, "label": target}


class LoadMisclassificationDataset(torch.utils.data.Dataset):
    """
    Args:
            conf_file (str): Path to the npz files directory of max_pred_probs.
            pred_file (str): Path to the npz file directory of preds.
            label_file (str): Path to the npz file directory of labels.
    """
    def __init__(self, conf_file, pred_file, label_file):
        print("Initializing Misclassification Dataset...")
        # Load all npz files from the ID directory
        self.conf_scores = load_npz_files(conf_file)
        self.preds = load_npz_files(pred_file)
        self.labels = load_npz_files(label_file)

    def __len__(self):
        return len(self.conf_scores)

    def __getitem__(self, idx):
        conf_score = self.conf_scores[idx]
        label = self.labels[idx]
        pred = self.preds[idx]
        return {"scores": conf_score, "label": label, "pred": pred}


def log_category_chart(run, labels, metric_values, title):
    # Replace nans to zeros if exists
    metric_values[torch.isnan(metric_values)] = 0.0
    # Converting nan to zeros
    data = [[label, val] for (label, val) in zip(labels, metric_values)]
    table = wandb.Table(data=data, columns=["Class Name", title])
    run.log({f"{title}": wandb.plot.bar(table, "Class Name", title, title=f"{title}")})
    return run


def log_eval_metrics(run, scalar_metrics_to_log, per_category_metrics_to_log, labels, summary_metrics, figs, plots_save_dir):
    # Log Scalar Metrics
    for metric_name in scalar_metrics_to_log:
        if metric_name in summary_metrics:
            try:
                run.log({f"test_{metric_name}": summary_metrics[metric_name]})
            except Exception as e:
                print(f"[ERROR]: Failed to log test_{metric_name}. {e}")
    # Log Per Category Metrics as Bar charts
    for metric_name in per_category_metrics_to_log:
        if metric_name in summary_metrics:
            per_category_metric = summary_metrics[f"{metric_name}"]  # tensor([clsval1, clsval2, ....])
            run = log_category_chart(run, labels, per_category_metric, f"{metric_name}(Test)")
    # Log Figures
    if figs:
        for fig_name, fig in figs.items():
            # Save the figure to wandb.
            try:
                run.log({fig_name: fig})
            except Exception as e:
                print(f"[ERROR]: Failed to log {fig_name}. {e}")
            # Save the figure locally
            try:
                print(f"[INFO]: Trying to save the {fig_name} image")
                if "multiple_fig_roc_" in fig_name:
                    # os.makedirs(f"./test_eval_plots", exist_ok=True)
                    write_image(fig=fig, file=f"./test_eval_plots/{fig_name}.pdf", format="pdf")
                else:
                    os.makedirs(f"./test_eval_plots/{plots_save_dir}", exist_ok=True)
                    write_image(fig=fig, file=f"./test_eval_plots/{plots_save_dir}/{fig_name}.pdf", format="pdf")
            except Exception as e:
                print(f"Failed to save the {fig_name} plot at {plots_save_dir}: {e}")
    return run
