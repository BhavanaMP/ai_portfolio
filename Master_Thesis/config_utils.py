import sys
import secrets
import string
from datetime import datetime
import argparse
import warnings
from typing import Union, List, Optional, Dict

from dataclasses import dataclass, field
import yaml


def generate_run_name(length=7):
    # Generate a random string
    alphabet = string.ascii_lowercase + string.digits
    random_string = ''.join(secrets.choice(alphabet) for _ in range(length))
    # Get the current date and time
    current_datetime = datetime.now().strftime("%d%m%Y_%H%M%S")  # Eg: "19072024_142530"
    run_name = f"run_{current_datetime}_{random_string}"
    return run_name


@dataclass
class Config:
    # CUDA and DDP related args
    world_size: int = 0
    init_method: str = None      # "tcp://127.0.0.1:12355"
    backend: str = "nccl"

    # Project Args
    architecture: str = None          # Unet_Resnet101, FPN_MIT_B3, DLV3Plus_CONVNEXT_small
    type: str = "baseline_50_5"       # baseline_50_5, baseline_OHEM_50_5, mcdropout_50_5, mcdropout_OHEM_50_5, EDL, EDL_UNC, EDL_OHEM
    model_type: str = "baseline"      # MCD, EDL, baseline
    name: str = None                  # architecture_type without 50_5

    # for reproducibility
    seed: int = 1
    
    # Pretrain
    is_pretrain: bool = False
    pretrain_ckpt_save_loc: str = "./checkpoints/pretrain_checkpoints"

    # Training
    is_train: bool = False

    # Common for Training and Pretrain
    run_id: str = field(default_factory=lambda: generate_run_name())
    train_dataset_name: str = None  # "railsem19", "cityscapes"
    num_epochs: int = 50
    ignore_index: int = 255
    crop_size: int = 1024
    batch_size: int = None
    lr: float = None
    encoder: str = None
    decoder: str = None
    
    
    # Just for Training
    # Finetune
    is_finetune: bool = False
    pretrained_mdl_ckpt_path: str = None
    # Resume
    resume_training: bool = False
    resume_log_dir: str = None
    # MCD
    train_mcd: bool = False
    forward_passes: int = 0
    # EDL
    train_edl: bool = False
    use_uncertainty_in_edl: bool = False
    # Common for MCD and EDL and vanillas
    use_ohem: bool = False
    # Training checkpoints paths
    baseline_ckpt_save_loc: str = "./checkpoints/baseline_checkpoints"
    mcd_ckpt_save_loc: str = "./checkpoints/mcd_checkpoints"
    edl_ckpt_save_loc: str = "./checkpoints/edl_checkpoints"

    # Test
    is_test_eval: bool = False
    eval_model_type: str = None  # "mcd", "edl", "baseline" Note: baseline runs from mcd model initialization itself.
    # PostHoc Calibration
    run_post_hoc_calib: bool = False
    use_tempsscaling: bool = False
    temp_init_val: float = 1.5
    posthoc_epochs: int = 20
    saved_temperature_path: str = f"./posthoc/temperature_checkpoints/{name}_{run_id}_temperature_scaling.json"
    # Railsem19 robustness
    run_robustness_inference: bool = False
    robustness_data_path_hf: str = None
    # Prediction
    is_predict: bool = False
    choose_random: bool = False
    plot_individual: bool = False
    selected_test_image:  Union[str, None] = None
    pred_img_path: str = None
    run_mcd_for_preds: str = None
    fig_title: str = None
    # OOD
    run_ood_inference: bool = False
    # MCDropout Inference
    run_mcdropout_inference: bool = False
    # Common for whole Test
    eval_dataset_name: str = None   # not considered OOD and AURRC, needed for ece and segmetrics
    model_path: str = None
    model_names: str = None

    # Calculate Eval Metrics Related Args
    calc_eval_metrics: bool = False
    calc_seg_metrics: bool = False
    tests_save_dir: str = None
    calc_ece: bool = False
    ece_base_dir: str = None
    ece_bins: int = None
    calc_aurrc: bool = False
    aurrc_base_dir: str = None
    aurrc_bins: str = None
    calc_ood_metrics: bool = False
    is_ood_multiple: bool = False
    multiple_ood_dirs: Optional[List[Dict[str, str]]] = field(default=None)
    ood_dir: str = None
    ood_name: str = None
    recall_level: float = None
    calc_misclassification: bool = False
    misclassification_dir: str = None
    misclassification_name: str = None
    misclass_recall_level:float = None
    plot_id_ood_distributions: bool = None
    baseline_ood_dir: str = None
    improv_ood_dir: str = None
    improv_mdl_title: str = None
    baseline_mdl_title: str = None

    def __post_init__(self):
        if not any([self.is_pretrain, self.is_predict, self.is_test_eval, self.is_train, self.calc_eval_metrics]):
            raise ValueError("Atleast any of is_pretrain, is_predict / is_test_eval / is_train should be True")


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


def merge_configs(config: Config, args: argparse.Namespace) -> Config:
    """
    Merges the configurations from the file and command-line arguments, giving precedence to command-line arguments.
    Default values are set in the dataclass.
    YAML files provide an easy way to override these defaults without modifying the code.
    Command-line arguments can further override the configurations defined in the YAML file.
    """
    # Override configuration with command-line arguments if they are provided
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config


def config_to_string(config: Config) -> str:
    config_str = "\n".join([f"{key}: {value}" for key, value in config.__dict__.items()])
    return config_str

def parse_multiple_ood_dirs(raw_list: List[str]) -> List[Dict[str, str]]:
    """
    Converts a list of alternating paths and names into a list of dictionaries for multiple OOD dirs.
    Example Input: ["./path1", "OOD_1", "./path2", "OOD_2"]
    Example Output: [{"path": "./path1", "name": "OOD_1"}, {"path": "./path2", "name": "OOD_2"}]
    """
    print(raw_list)
    if len(raw_list) % 2 != 0:
        raise ValueError("Paths and names must be provided in pairs (e.g., ./path1 Baseline).")
    # wrong.. handle it later when you need to pass form cmd
    return [{"path": raw_list[i], "name": raw_list[i+1]} for i in range(0, len(raw_list), 2)]


def parse_arguments():
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="Bhavana Master Thesis")

    # Custom action to parse image paths
    class ImagePathsAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Convert second value to None if it's the string "None"
            setattr(namespace, self.dest, (values[0], None if values[1] == "None" else values[1]))

    # Custom action to parse image paths
    class ModelNamesAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Ensure the values are stored as a tuple
            if len(values) != 2:
                parser.error(f"{option_string} requires exactly two arguments. The first is encoder name and the second is decoder name.")
            setattr(namespace, self.dest, (values[0], values[1]))

    class StoreTrueOrNoneAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, required=False, help=None):
            super(StoreTrueOrNoneAction, self).__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=0,  # This action does not consume any arguments
                default=default,
                required=required,
                help=help
            )

        def __call__(self, parser, namespace, values, option_string=None):
            # If the flag is present, set the destination attribute to True
            setattr(namespace, self.dest, True)

    # DDP Args
    parser.add_argument("--world_size", type=int, help="Number of processes for distributed training")
    parser.add_argument('--init_method', type=str)
    parser.add_argument("--backend", type=str, help="Distributed communications Backend")
    parser.add_argument("--seed", type=int, help="seed for initializing training.")
    parser.add_argument("--config_file", type=str, help="Path to the yaml config file")
    
    # Pretraining Args
    parser.add_argument("--is_pretrain", action=StoreTrueOrNoneAction, help="Start Training the Selected Model")
    parser.add_argument("--pretrain_ckpt_save_loc", type=str, help="Local Save location of pretrained model checkpoint")
    
    # Training Args
    parser.add_argument("--is_train", action=StoreTrueOrNoneAction, help="Start Training the Selected Model")
    
    # Common Args for Train and Pretrain
    parser.add_argument("--encoder", type=str,  choices=["resnet101", "mit_b3", "tu-convnext_small"], help="Choose the encoder: resnet101, mit_b3, tu-convnext_small")  # resnet101, mit_b3, tu-convnext_small
    parser.add_argument("--decoder", type=str, choices=["Unet", "FPN", "DeepLabV3Plus"], help="Choose the decoder: FPN, Unet, DeepLabV3Plus")  # FPN, Unet, DeepLabV3Plus
    parser.add_argument("--num_epochs", type=int, help="the number of epochs to train for")
    parser.add_argument("--lr", type=float, help="learning rate to use for model")
    parser.add_argument("--crop_size", type=int, help="Crop size tp use for training")
    parser.add_argument("--ignore_index", type=int, help="Index to be ignored while training")
    parser.add_argument("--batch_size", type=int, help="number of samples per batch")  # 6 for convnext with predunc# 3 for fpnmitb3 #24 for posthocalib val# 8 for ur101 mcd 5 ohem # 6 when using ohem+mcdrput+weights 

    # Just for Training
    # Finetune Training Args
    parser.add_argument("--is_finetune", action=StoreTrueOrNoneAction, help="Whether you wanna finetune the mdoel with pretrained weights")
    parser.add_argument("--pretrained_mdl_ckpt_path", type=str, help="Saved Pretrained Checkpoint Path")
    # Resume Training Args
    parser.add_argument("--resume_training", action=StoreTrueOrNoneAction, help="Whether you wanna resume training")
    parser.add_argument("--resume_log_dir", type=str, help="Same Log dir of the run which you want to resume.")
    # MCD Model Args
    parser.add_argument("--train_mcd", action=StoreTrueOrNoneAction, help="Train the MCD Model i.e whether to use MCD Uncertainty Weighting in MCD Model")
    parser.add_argument("--forward_passes", type=int, help="No. of forward passes for MCD Training")
    # EDL Model Args
    parser.add_argument("--train_edl", action=StoreTrueOrNoneAction, help="Train the EDL Model")
    parser.add_argument("--use_uncertainty_in_edl", action=StoreTrueOrNoneAction, help="Whether to use uncertainty weighting in EDL")
    # Common for MCD, EDL and vanillas
    parser.add_argument("--use_ohem", action=StoreTrueOrNoneAction, help="Whether to enable OHEM")
    # Training Checkpoint Saving Args
    parser.add_argument("--train_dataset_name", type=str, choices=["railsem19, cityscapes"], help="Give the dataset name that is to be trained on.")
    parser.add_argument("--baseline_ckpt_save_loc", type=str, help="Local Save location of trained baseline model checkpoint")
    parser.add_argument("--mcd_ckpt_save_loc", type=str, help="Local Save location of trained MCD model checkpoint")
    parser.add_argument("--edl_ckpt_save_loc", type=str, help="Local Save location of trained EDL model checkpoint")
    # Test Evaluation Args - For test, we only consider cmd line args.
    parser.add_argument("--is_test_eval", action=StoreTrueOrNoneAction, help="Run Test Set evaluation using trained model. Make sure to give right dataset name")
    parser.add_argument("--eval_model_type", type=str, choices=["mcd, edl, baseline"], help="Give the model type. This helps in selected the respective model during statedict loading.")
    # Post-Hoc Calibration
    parser.add_argument("--run_post_hoc_calib", action=StoreTrueOrNoneAction, help="post hoc calibration for baseline model")
    parser.add_argument("--use_tempsscaling", action=StoreTrueOrNoneAction, help="Apply temp scaling for baseline model Test Evaluation")
    parser.add_argument("--temp_init_val", type=float, help="Initial Temperature value for temp scaling calibration.")
    parser.add_argument("--posthoc_epochs", type=int, help="Num of epochs to train the temperature parameter post hoc calibration.")
    parser.add_argument("--saved_temperature_path", type=str, help="Path of saved temperature json file")  # Eg: ./posthoc_calibration/temperature_checkpoints/{args.config.run_name}_{args.config.run_id}_temperature_scaling.json
    # Robustness Test Evaluation
    parser.add_argument("--run_robustness_inference", action=StoreTrueOrNoneAction, help="Inference on Robustness Test sets of just railsem19 test set")
    parser.add_argument("--robustness_data_path_hf", type=str, help="Huggingface path of several weather pertubations of railsem19")
    # Prediction Args
    parser.add_argument("--is_predict", action=StoreTrueOrNoneAction, help="Run prediction on unseen image")
    parser.add_argument("--choose_random", action=StoreTrueOrNoneAction, help="Randomly choose the image from the test set for prediction.")
    parser.add_argument("--selected_test_image", help="Name of the image selected for prediction from the test dataset. This can be provided by the user or chosen randomly.")
    parser.add_argument("--pred_img_path", nargs=2, action=ImagePathsAction, metavar=("path1", "path2"), help="Path to unseen image and the ground truth. GT path can be None. Example: --pred_img_path path/to/image.jpg path/to/gt.jpg")
    parser.add_argument("--run_mcd_for_preds", action=StoreTrueOrNoneAction, help="Run MCD for unseen image prediction to get predictive uncertainty map")
    parser.add_argument("--fig_title", type=str, help="Prediction Figure Title")
    parser.add_argument("--plot_individual", action=StoreTrueOrNoneAction, help="Save the prediction plots in a directory as individual figures.")
    
    # Test MCD Inference
    parser.add_argument("--run_mcdropout_inference", action=StoreTrueOrNoneAction, help="MCD Inference on Test Datasets.")
    # OOD Test Related Args
    parser.add_argument("--run_ood_inference", action=StoreTrueOrNoneAction, help="Inference on OOD Test Datasets. Make sure to give right dataset name.")
    # Common Args for Test Evaluation, Prediction, Posthoc calibration, OOD Inference
    parser.add_argument("--eval_dataset_name", type=str, choices=["railsem19, cityscapes", "fishyscapes", "obstacles", "anomaly", "streethazards", "lostandfound", "lostandfoun_full"], help="Give the dataset name that is to be evaluated or predicted on. Also applies on ood. Transforms are applied based on this")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--model_names", nargs=2, action=ModelNamesAction, metavar=("ENCODER", "DECODER"), help="Specify the encoder and decoder model names. Example: --model_names resnet101 Unet")
    # Calculate Eval Metrics related Args
    parser.add_argument("--calc_eval_metrics", action=StoreTrueOrNoneAction, help="Flag that lets us to calculate eval metrics of test sets.")

    parser.add_argument("--calc_seg_metrics", action=StoreTrueOrNoneAction, help="Flag that lets calculate segmentation metrics of test sets. Seg Metrics are mIoU, OverallAccuracy, nll, ece per img, iou per img, acc per img, per category iou, acc, entropy, unc, edl_belief, edl_unc")
    parser.add_argument("--tests_save_dir", type=str, help="Base dir path of where the test results are saved. Examples: path/to/Model/ <Original / MCDInference / OOD_<Datasetname> / OOD_<Datasetname>/MCDInference / Robustness_<test_name> >")

    parser.add_argument("--calc_ece", action=StoreTrueOrNoneAction, help="Flag that lets to calculate ECE of test sets. We do these separately because we need to load the entire test dataset saved metrics into memory")
    parser.add_argument("--ece_base_dir", type=str, help="Base dir path that contains subdirs of where the test set probs and labels results are saved. Examples: path/to/Model/< Original / MCDInference / OOD_<Datasetname> / MCDInference Robustness_<test_name> >")
    parser.add_argument("--ece_bins", type=int, help="Number of bins to consider while calculating the ECE Metric.")

    parser.add_argument("--calc_aurrc", action=StoreTrueOrNoneAction, help="Flag that lets to calculate AURRC of test sets. We do these separately because we need to load the entire test dataset saved metrics into memory")
    parser.add_argument("--aurrc_base_dir", type=str, help="Base dir path that contains subdirs of where the test set preds and labels results are considered. Examples: path/to/Model/< Original / MCDInference / OOD_<Datasetname> / MCDInference Robustness_<test_name> >")
    parser.add_argument("--aurrc_bins", type=int, help="Number of bins to consider while calculating the AURRC Metric.")

    parser.add_argument("--calc_ood_metrics", action=StoreTrueOrNoneAction, help="Flag that lets to calculate OOD Metrics(FPR@thresh, AUROC, AUPR) of OOD datasets. We do these separately because we need to load the entire test dataset saved metrics into memory")
    parser.add_argument("--is_ood_multiple", action=StoreTrueOrNoneAction, help="Flag that lets to calculate OOD Metrics(FPR@thresh, AUROC, AUPR) of  multilple OOD datasets at once.")
    parser.add_argument("--multiple_ood_dirs", nargs="+", help="Specify the multiple OOD paths.")
    parser.add_argument("--ood_dir", type=str, help="Base dir path that contains subdirs of where the OOD dataset max_pred_probs and labels results are considered. Examples: path/to/Model/<OOD_<Datasetname> or OOD_<Datasetname>/MCDInference>")
    parser.add_argument("--ood_name", type=str, help="Name of the OOD Dataset. Make sure it matches with OOD directory name")
    parser.add_argument("--recall_level", type=float, help="Threshold to calculate FPR@recall_level")
    
    parser.add_argument("--calc_misclassification", action=StoreTrueOrNoneAction, help="Flag that lets to calculate calc_misclassification Metrics(FPR@thresh, AUROC, AUPR) of datasets. We do these separately because we need to load the entire test dataset saved metrics into memory")
    parser.add_argument("--misclassification_dir", type=str, help="Base dir path that contains subdirs of where the dataset max_pred_probs and labels results are considered. Examples: path/to/Model/<Datasetname> or <Datasetname>/MCDInference>")
    parser.add_argument("--misclassification_name", type=str, help="Name of the Dataset. Make sure it matches with directory name")
    parser.add_argument("--misclass_recall_level", type=float, help="Threshold to calculate FPR@recall_level")

    parser.add_argument("--plot_id_ood_distributions", action=StoreTrueOrNoneAction, help="When set, Plot OOD related plots like enotropy distribution of baseline and the improved model as violin plots by loading whole test dataset results")
    parser.add_argument("--baseline_ood_dir", type=str, help="Base directory path of OOD results of a baseline model. Code will look for entropies and labels subdirs from this baseline_ood_dir.")
    parser.add_argument("--improv_ood_dir", type=str, help="Base directory path of OOD results of a Improved or proposed model. Code will look for entropies and labels subdirs from this improv_ood_dir.")
    parser.add_argument("--improv_mdl_title", type=str, help="Name of the improved model. Examples: MCD, MCD_OHEM, EDL, EDL_OHEM, EDL_UNC")
    parser.add_argument("--baseline_mdl_title", type=str, help="Name of the baseline model. Examples: Baseline")
    
    # Parse Args
    args = parser.parse_args(sys.argv[1:])  # sys.argv[0] is the python file itself

    # Load config from file.
    config_file = load_config(args.config_file)
    args = merge_configs(config_file, args)

    # Argument validation checks
    if (args.is_predict or args.is_test_eval) and not args.model_path and not args.model_names:
        # Make sure model path exists when its prediction
        parser.error("Both '--model_path' and '--model_names' are required for prediction or test set evaluation")

    if args.resume_training and not args.resume_log_dir:
        # Make sure resume_log_dir exists when the training is being resumed
        parser.error("--resume_log_dir is required when --resume_training is set")
    
    if args.is_finetune and not args.is_train:
        parser.error("--is_train must be set when --is_finetune is set")
        
    # # Process the multiple OOD dirs - Handle later when you needed it to pass from cmd
    # if args.calc_ood_metrics:
    #     if args.is_ood_multiple and args.multiple_ood_dirs:
    #         args.multiple_ood_dirs = parse_multiple_ood_dirs(args.multiple_ood_dirs)

    return args
