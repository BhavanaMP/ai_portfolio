import os
from datetime import timedelta
import gc

import torch
# Distributed Training Imports DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.distributed import init_process_group, destroy_process_group
import wandb

from datasets import disable_caching

from config_utils import parse_arguments

from modelling.pretrain import AutoEncoderPreTrainer
from modelling.mcd import MCDTrainer
from modelling.edl import EDLTrainer

from evaluation.evaluation_inference import Evaluation
from evaluation.calc_eval_metrics import TestMetrics
from posthoc.temperature_calibration import TemperatureScaler


def ddp_setup(rank: int, world_size: int, backend, init_method):
    """
    Args:
    rank: Unique identifier of each process. Each GPU has one process
    world_size: Total No. of processes or GPUs
    backend: Backend to use for distributed communcation among GPUs
    (nccl - NVIDIA Collective Communications Library)
    """
    # Initiates the group of our process for them to see each other, we launch from process 0 i.e rank 0,, and since we gave local host as masteraddr, that paritcular gpu from where we launch the process will be our master GPU reposnible for communications
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    # torch.cuda.set_device(rank)  # 
    torch.cuda.device(rank)
    init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank, timeout=timedelta(seconds=3600)) # 6000 = 100 minutes
    dist.barrier()


def main(rank, args, run=None):
    """
    Note: wandb run is initialized only for test evaluations.

    We use tensorboard for tracking training progress.
    """
    print(f"[INFO]: Rank initialized: {rank}")
    # Set up and Initialize the process group
    if args.is_distributed:
        ddp_setup(rank=rank, world_size=args.world_size, backend=args.backend, init_method=args.init_method)
    # Sets device=cuda:0 explicilty if gpu exist & only 1 available
    elif torch.cuda.is_available() and args.world_size == 1:
        rank = 0
    else:
        rank = "cpu"
    
    if args.is_pretrain:
        print("[INFO]: Pre Training the Model")
        pretrainer = AutoEncoderPreTrainer(args, rank)
        pretrainer.train()
    elif args.is_train:
        # Training
        print("[INFO]: Training the Model")
        if args.is_train:
            # For baseline, we will just use MCDTrainer that can run baseline model.
            trainer = MCDTrainer(args, rank)
            if args.train_mcd:
                trainer = MCDTrainer(args, rank)
            elif args.train_edl:
                trainer = EDLTrainer(args, rank)
        # Launch Training
        train_results = trainer.train()
    elif args.run_post_hoc_calib:
        # Post-hoc Calibration
        print("[INFO]: Running Post-hoc temperature scaling calibration")
        scaler = TemperatureScaler(mdl_path=args.model_path, mdl_names=args.model_names, batch_size=args.batch_size, lr=args.lr, device=rank, init_val=args.temp_init_val, epochs=args.posthoc_epochs, saved_temperature_path=args.saved_temperature_path)
        _, optimal_temperature = scaler.fit()
        print("Optimal temperature", optimal_temperature)
    elif args.calc_eval_metrics:
        calc_metrics = TestMetrics(args, run, rank)
        _, _, run = calc_metrics.main()
    else:
        evaluator = Evaluation(args, run, rank)
        if args.is_test_eval:
            # Test Evaluation
            print("[INFO]: Running Test Evaluation")
            evaluator.test_eval()
        elif args.is_predict:
            # Prediction
            print("[INFO]: Running Prediction")
            evaluator.predict(inputs=args.pred_img_path, choose_random=args.choose_random, plot_individual=args.plot_individual, selected_test_image=args.selected_test_image, run_mcd_inference=args.run_mcd_for_preds, model_type=args.model_type, fig_title=args.fig_title)
    # Destroy the process group
    if args.is_distributed:
        destroy_process_group()


# Main
if __name__ == "__main__":
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    # Get the Args
    args = parse_arguments()
    
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
    args.is_distributed = (args.world_size >= 1)
    print(f"\nARGS: {args}")
    # Set up respective run logger
    if args.is_train or args.is_test_eval or args.is_pretrain:
        # For training, we use tensorbaord. We dont need wandb run for just test results saving.
        _run = None
    elif args.is_predict or args.calc_eval_metrics:
        # Log Test Evaluation Results or Predictions Results to Wandb
        try:
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
        except Exception as e:
            print(f"Wandb Login Failed: {e}")
        # Wandb set up
        wandb.require("service")

        wandb_project_name = "Prediction" if args.is_predict else "Test Evaluation"
        wandb_config = dict(architecture=f"{args.architecture}", type=f"{args.type}", model_type=f"{args.model_type}",
                            name=f"{args.name}", dataset_name=f"{args.eval_dataset_name}", epochs=args.num_epochs,
                            batch_size=1, learning_rate=args.lr, project_name=wandb_project_name)
        _run = wandb.init(name=wandb_config["name"],  project=wandb_project_name, config=wandb_config,
                          entity=os.environ.get("WANDB_ENTITY"), group="DDP",
                          resume=True, mode="offline")
        rdir = os.path.split(_run.dir)[0]

    # Initiate the Multiprocesses
    try:
        print("[INFO]: Spawning")
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        print("[WARNING]: Spawning error")
        pass
    processes = []
    for rank in range(args.world_size):
        p = Process(target=main, args=(rank, args, _run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if args.is_predict or args.calc_eval_metrics:
        # finish the wandb run
        _run.finish()
        # os.system('find /scratch2/malla/tmp -type d -user malla -ls -name "*wandb*" -print -maxdepth 1 -exec rm -rf {} \;')
        print("[INFO] Run Finished....")

    if args.is_train:
        print("[INFO] Train Run Finished....")

    if args.is_test_eval:
        print("[INFO] Test Run Finished....")
        
    if args.is_pretrain:
        print("[INFO] Pretrain Run Finished....")
        
