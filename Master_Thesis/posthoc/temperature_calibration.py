from typing import Literal, Tuple

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from datasets import load_dataset

from tqdm import tqdm

from evaluation.evaluation_utils import load_model_for_evaluation
from data.prepare_railsem19 import CustomRailsem19Dataset, get_railsem19_transforms, get_railsem19_labels

from posthoc.lbfgsnew import LBFGSNew
from posthoc.posthoc_utils import save_temperature


class Scaler(nn.Module):
    trained = False

    def __init__(self, mdl_path: str, mdl_names: Tuple[str, str],
                 batch_size: int, lr: float = 0.01, max_iter: int = 50, patience: int = 5,
                 epochs: int = 1, device=None, saved_temperature_path="./temperature_scaling.json") -> None:
        """Virtual class for scaling post-processing for calibrated probabilities.

        Args:
            mdl_path (str): Path to Model(nn.Module) that is to be calibrates.
            mdl_names tuple(str, str): Names of encoder and decoder to load the model from smp.
            batch_size (int): batch_size for fitting the scaler.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
            max_iter (int, optional): Maximum number of iterations for the
                optimizer. Defaults to 100.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use
                for optimization. Defaults to None.

        Reference:
            Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. On calibration
            of modern neural networks. In ICML 2017.
            # Ref: https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/main/README.md
        """
        super().__init__()
        self.device = device
        self.models_paths = mdl_path
        self.models_names = mdl_names
        self.epochs = epochs
        self.patience = patience
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        print(f"[INFO]: Loading the model from path {self.models_paths}, {self.models_names}")
        self.id2label, _, _ = get_railsem19_labels()
        self.model = load_model_for_evaluation(self.models_paths, self.models_names, id2label=self.id2label, model_type="mcd")
        self.freeze_model_parameters()  # Freeze before setting temperature
        self.model.to(self.device)
        self.batch_size = batch_size
        self.calibration_set = self.load_calib_ds()
        self.saved_temperature_path = saved_temperature_path

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.lr = lr

        if max_iter <= 0:
            raise ValueError("Max iterations must be positive.")
        self.max_iter = int(max_iter)

    def freeze_model_parameters(self):
        print("[INFO]: Freezing the Model parameters")
        for name, param in self.model.named_parameters():
            # if name == "temperature":
            #     print("[INFO]: Temperature Found")
            # if name != "temperature":  # Check parameter name
                param.requires_grad = False

    def load_calib_ds(self):  # FIXME: LOading the dataset based on dataset name
        print("[INFO]: Extracting Railsem19 Calib dataset from hf")
        calib_ds = load_dataset("BhavanaMalla/railsem19_val_split")
        calib_dataset = calib_ds["val"]
        print(f"[INFO]: Total Calib images: {len(calib_dataset)}")
        
        transforms = get_railsem19_transforms()
        calib_split = CustomRailsem19Dataset(calib_dataset, transforms, split="val")
        return calib_split
    
    @torch.no_grad()
    def get_logits(self, inputs):
        self.model.eval()  # set model to eval mode (stops BN, dropout)
        logits = self.model(inputs)  # bs, num_classes, h, w
        return logits
        

    def fit(self, save_logits: bool = False, progress: bool = True,) -> "Scaler":
        """Fit the temperature parameters to the calibration data.
        Args:
            save_logits (bool, optional): Whether to save the logits and
                labels. Defaults to False.
            progress (bool, optional): Whether to show a progress bar.
                Defaults to True.
        Returns:
            Scaler: Calibrated scaler.
        """
        print("[INFO]: Fitting the Temperature Scaler")
        sampler = DistributedSampler(self.calibration_set) if dist.is_available() and dist.is_initialized() else None
        calibration_dl = DataLoader(
            self.calibration_set, batch_size=self.batch_size, pin_memory=True,
            drop_last=False, shuffle=True, sampler=sampler
        )

        best_loss = float('inf')  # Initialize best_loss with a high value
        best_model_state = None   # Initialize best model state
        epochs_no_improve = 0     # Counter for early stopping

        # LBFGS with batch mode. Pytorch doesnt support batch mode.
        optimizer = LBFGSNew(params=self.temperature, lr=self.lr,
                             max_iter=self.max_iter, batch_mode=True)

        self.train()
        for epoch in tqdm(range(self.epochs)):
            if dist.is_available() and dist.is_initialized():
                calibration_dl.sampler.set_epoch(epoch)
            
            running_loss = 0.0
            for _, batch in enumerate(tqdm(calibration_dl, disable=not progress)):
                inputs = batch["image"].to(self.device)  # bs, 3, h, w
                labels = batch["mask"].to(self.device)  # bs, 1, h, w
                logits = self.get_logits(inputs)  # bs, num_classes, h, w
                num_classes = logits.shape[1]
                # Reshape and detach
                logits = logits.permute(0, 2, 3, 1)  # bs, h, w, num_classes
                logits = logits.view(-1, num_classes).detach()  # bs * h * w, num_classes
                labels = labels.squeeze(dim=1)  # bs, h, w
                labels = labels.view(-1).detach()  # bs * h * w
                del inputs
                
                def calib_eval() -> float:
                    optimizer.zero_grad()
                    loss = self.criterion(self._scale(logits), labels)
                    # if loss.requires_grad:
                    #     print("[INFO]: Inside the requires grad loop")
                    loss.backward()
                    return loss
                optimizer.step(calib_eval)

                # calculate the loss again for monitoring
                loss = calib_eval()
                running_loss += loss.item()
            
            average_loss = running_loss / len(calibration_dl)
            print(f"[INFO]: Epoch: {epoch} | Running Loss: {average_loss}")

            # Save the model if the current running loss is lower than the best loss
            if average_loss < best_loss:
                best_loss = average_loss
                print(f"Optimal temperature: {self.temperature[0]}")
                if hasattr(self, "module"):
                    best_model_state = self.module.state_dict().copy()
                else:
                    best_model_state = self.state_dict().copy()
                print(f"[INFO]: Best model saved with loss: {best_loss}")
                epochs_no_improve = 0  # Reset counter if improvement
            else:
                epochs_no_improve += 1  # Increment counter if no improvement
                print(f"[INFO]: No improvement for {epochs_no_improve} epoch(s)")
            
            # Early stopping condition
            if epochs_no_improve >= self.patience:
                print(f"[INFO]: Early stopping triggered after {epoch+1} epochs")
                break

        self.trained = True

        # Load the best model state after training
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"[INFO]: Loaded best model with loss: {best_loss}")
        # Save the temperature
        save_temperature(self.temperature[0].detach().cpu().item(), file_path=f"{self.saved_temperature_path}")
        return self, self.temperature[0].detach().cpu().item()

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        if not self.trained:
            print(
                "TemperatureScaler has not been trained yet. Returning "
                "manually tempered inputs."
            )
        return self._scale(self.get_logits(inputs))

    def _scale(self, logits: Tensor) -> Tensor:
        """Scale the logits with the optimal temperature.

        Args:
            logits (Tensor): Logits to be scaled.

        Returns:
            Tensor: Scaled logits.
        """
        raise NotImplementedError

    def fit_predict(self, calibration_set: Dataset, progress: bool = True,) -> Tensor:
        self.fit(calibration_set, save_logits=True, progress=progress)
        return self(self.logits)

    @property
    def temperature(self) -> list:
        raise NotImplementedError


class TemperatureScaler(Scaler):
    def __init__(
        self,
        mdl_path: str,
        mdl_names: Tuple[str, str],
        batch_size: int,
        init_val: float = 1,
        lr: float = 0.1,
        max_iter: int = 100,
        epochs: int = 1,
        device=None,
        saved_temperature_path="./temperature_scaling.json"
    ) -> None:
        """Temperature scaling post-processing for calibrated probabilities.

        Args:
            model (nn.Module): Model to calibrate.
            init_val (float, optional): Initial value for the temperature.
                Defaults to 1.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
            max_iter (int, optional): Maximum number of iterations for the
                optimizer. Defaults to 100.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use
                for optimization. Defaults to None.

        Reference:
            Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. On calibration
            of modern neural networks. In ICML 2017.
        """
        super().__init__(mdl_path=mdl_path, mdl_names=mdl_names,
                         batch_size=batch_size, lr=lr, max_iter=max_iter,
                         epochs=epochs, device=device, saved_temperature_path=saved_temperature_path)

        if init_val <= 0:
            raise ValueError("Initial temperature value must be positive.")

        self.set_temperature(init_val)
        
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO]: Number of trainable parameters: {total_trainable_params}")


    def set_temperature(self, val: float) -> None:
        """Set the temperature to a fixed value.

        Args:
            val (float): Temperature value.
        """
        if val <= 0:
            raise ValueError("Temperature value must be positive.")

        self.temp = nn.Parameter(torch.ones(1, device=self.device) * val, requires_grad=True)
        # Register the temperature parameter with a specific name to add it to the model params
        self.register_parameter('temperature', self.temp)

    def _scale(self, logits: Tensor) -> Tensor:
        """Scale the prediction with the optimal temperature.

        Args:
            logits (Tensor): logits to be scaled.

        Returns:
            Tensor: Scaled logits.
        """
        return logits / self.temperature[0]

    @property
    def temperature(self) -> list:
        return [self.temp]
