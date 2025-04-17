import random
import os
import torch
import numpy as np


def set_seeds(seed):
    """
    Setting seed for reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def print_trainable_parameters(model, print_msg):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"\n[INFO]:-----{print_msg}......")
    print(f"trainable params: {trainable_params / 1e+6:.2f} M || all params: {all_param / 1e+6:.2f} M || trainable%: {100 * trainable_params / all_param:.2f} %")
