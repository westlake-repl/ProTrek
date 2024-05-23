import torch
import abc
import os
import copy

import pytorch_lightning as pl
from utils.lr_scheduler import *
from torch import distributed as dist


class AbstractModel(pl.LightningModule):
    def __init__(self,
                 lr_scheduler_kwargs: dict = None,
                 optimizer_kwargs: dict = None,
                 save_path: str = None,
                 from_checkpoint: str = None,
                 load_prev_scheduler: bool = False,
                 save_weights_only: bool = True,):
        """

        Args:
            lr_scheduler: Kwargs for lr_scheduler
            optimizer_kwargs: Kwargs for optimizer_kwargs
            save_path: Save trained model
            from_checkpoint: Load model from checkpoint
            load_prev_scheduler: Whether load previous scheduler from checkpoint
            load_strict: Whether load model strictly
            save_weights_only: Whether save only weights or also optimizer and lr_scheduler
            
        """
        super().__init__()
        self.initialize_model()
        
        self.metrics = {}
        for stage in ["train", "valid", "test"]:
            stage_metrics = self.initialize_metrics(stage)
            # Rigister metrics as attributes
            for metric_name, metric in stage_metrics.items():
                setattr(self, metric_name, metric)
                
            self.metrics[stage] = stage_metrics
        
        if lr_scheduler_kwargs is None:
            # Default lr_scheduler
            self.lr_scheduler_kwargs = {
                "class": "ConstantLRScheduler",
                "init_lr": 0,
            }
            print("No lr_scheduler_kwargs provided. The default learning rate is 0.")

        else:
            self.lr_scheduler_kwargs = lr_scheduler_kwargs
        
        if optimizer_kwargs is None:
            # Default optimizer
            self.optimizer_kwargs = {
                "class": "AdamW",
                "betas": (0.9, 0.98),
                "weight_decay": 0.01,
            }
            print("No optimizer_kwargs provided. The default optimizer is AdamW.")
        else:
            self.optimizer_kwargs = optimizer_kwargs
        self.init_optimizers()

        self.save_path = save_path
        self.save_weights_only = save_weights_only
        
        # temp_step is used for accumulating gradients
        self.temp_step = 0
        self.step = 0
        self.epoch = 0
        
        self.load_prev_scheduler = load_prev_scheduler
        self.from_checkpoint = from_checkpoint
        if from_checkpoint:
            self.load_checkpoint(from_checkpoint)

    @abc.abstractmethod
    def initialize_model(self) -> None:
        """
        All model initialization should be done here
        Note that the whole model must be named as "self.model" for model saving and loading
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward propagation
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def initialize_metrics(self, stage: str) -> dict:
        """
        Initialize metrics for each stage
        Args:
            stage: "train", "valid" or "test"
        
        Returns:
            A dictionary of metrics for the stage. Keys are metric names and values are metric objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_func(self, stage: str, outputs, labels) -> torch.Tensor:
        """

        Args:
            stage: "train", "valid" or "test"
            outputs: model outputs for calculating loss
            labels: labels for calculating loss

        Returns:
            loss

        """
        raise NotImplementedError

    @staticmethod
    def load_weights(model, weights):
        model_dict = model.state_dict()

        unused_params = []
        missed_params = list(model_dict.keys())

        for k, v in weights.items():
            if k in model_dict.keys():
                model_dict[k] = v
                missed_params.remove(k)

            else:
                unused_params.append(k)

        if len(missed_params) > 0:
            print(f"\033[31mSome weights of {type(model).__name__} were not "
                  f"initialized from the model checkpoint: {missed_params}\033[0m")

        if len(unused_params) > 0:
            print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

        model.load_state_dict(model_dict)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure=None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        self.temp_step += 1
        if self.temp_step == self.trainer.accumulate_grad_batches:
            self.step += 1
            self.temp_step = 0
    
    # For pytorch-lightning 1.9.5
    # def optimizer_step(
    #     self,
    #     epoch: int,
    #     batch_idx: int,
    #     optimizer,
    #     optimizer_idx: int = 0,
    #     optimizer_closure=None,
    #     on_tpu: bool = False,
    #     using_native_amp: bool = False,
    #     using_lbfgs: bool = False,
    # ) -> None:
    #     super().optimizer_step(
    #         epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs
    #     )
    #     self.temp_step += 1
    #     if self.temp_step == self.trainer.accumulate_grad_batches:
    #         self.step += 1
    #         self.temp_step = 0

    def on_train_epoch_end(self):
        self.epoch += 1
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.98))
        # for _ in range(1000):
        #     outputs = self(**inputs)
        #     loss = self.loss_func('train', outputs, labels)
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #
        # raise
        
        outputs = self(**inputs)
        loss = self.loss_func('train', outputs, labels)
        
        self.log("loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        loss = self.loss_func('valid', outputs, labels)
        self.valid_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        
        loss = self.loss_func('test', outputs, labels)
        self.test_outputs.append(loss)
        return loss
    
    def on_train_start(self) -> None:
        # Load previous scheduler
        if getattr(self, "prev_schechuler", None) is not None:
            try:
                self.step = self.prev_schechuler["global_step"]
                self.epoch = self.prev_schechuler["epoch"]
                self.best_value = self.prev_schechuler["best_value"]
                self.lr_scheduler.load_state_dict(self.prev_schechuler["lr_scheduler"])
                print(f"Previous training global step: {self.step}")
                print(f"Previous training epoch: {self.epoch}")
                print(f"Previous best value: {self.best_value}")
                print(f"Previous lr_scheduler: {self.prev_schechuler['lr_scheduler']}")
                
                # Load optimizer state
                if hasattr(self.trainer.strategy, "deepspeed_engine"):
                    # For DeepSpeed strategy
                    try:
                        self.trainer.strategy.deepspeed_engine.load_checkpoint(self.from_checkpoint)
                    except Exception as e:
                        print(e)

                else:
                    # For DDP strategy
                    self.optimizer.load_state_dict(self.prev_schechuler["optimizer"])

            except Exception as e:
                print(e)
                raise Exception("Error in loading previous scheduler. Please set load_prev_scheduler=False")
    
    def on_validation_epoch_start(self) -> None:
        setattr(self, "valid_outputs", [])
    
    def on_test_epoch_start(self) -> None:
        setattr(self, "test_outputs", [])
            
    def load_checkpoint(self, from_checkpoint: str) -> None:
        """
        Args:
            from_checkpoint:  Path to checkpoint.
        """
        
        # If ``from_checkpoint`` is a directory, load the checkpoint in it
        if os.path.isdir(from_checkpoint):
            basename = os.path.basename(from_checkpoint)
            from_checkpoint = os.path.join(from_checkpoint, f"{basename}.pt")

        state_dict = torch.load(from_checkpoint, map_location=self.device)
        self.load_weights(self.model, state_dict["model"])
        
        if self.load_prev_scheduler:
            state_dict.pop("model")
            self.prev_schechuler = state_dict
        
    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True) -> None:
        """
        Save model to save_path
        Args:
            save_path: Path to save model
            save_info: Other info to save
            save_weights_only: Whether only save model weights
        """
        dir = os.path.dirname(save_path)
        os.makedirs(dir, exist_ok=True)
        
        state_dict = {} if save_info is None else save_info
        state_dict["model"] = self.model.state_dict()
        
        # Convert model weights to fp32
        for k, v in state_dict["model"].items():
            state_dict["model"][k] = v.float()
            
        if not save_weights_only:
            state_dict["global_step"] = self.step
            state_dict["epoch"] = self.epoch
            state_dict["best_value"] = getattr(self, f"best_value", None)
            state_dict["lr_scheduler"] = self.lr_schedulers().state_dict()
            
            # If not using DeepSpeed, save optimizer state
            if not hasattr(self.trainer.strategy, "deepspeed_engine"):
                state_dict["optimizer"] = self.optimizers().optimizer.state_dict()

        torch.save(state_dict, save_path)

    def check_save_condition(self, now_value: float, mode: str, save_info: dict = None) -> None:
        """
        Check whether to save model. If save_path is not None and now_value is the best, save model.
        Args:
            now_value: Current metric value
            mode: "min" or "max", meaning whether the lower the better or the higher the better
            save_info: Other info to save
        """

        assert mode in ["min", "max"], "mode should be 'min' or 'max'"

        if self.save_path is not None:
            # In case there are variables to be included in the save path
            save_path = eval(f"f'{self.save_path}'")
            
            dir = os.path.dirname(save_path)
            os.makedirs(dir, exist_ok=True)
            
            # Check whether to save model
            best_value = getattr(self, f"best_value", None)
            if best_value is not None:
                if mode == "min" and now_value >= best_value or mode == "max" and now_value <= best_value:
                    return
                
            setattr(self, "best_value", now_value)
                
            # For DeepSpeed strategy
            if hasattr(self.trainer.strategy, "deepspeed_engine"):
                if not self.save_weights_only:
                    self.trainer.strategy.deepspeed_engine.save_checkpoint(save_path, tag="deepspeed_ckpt")
                
                # Save a complete checkpoint
                if dist.get_rank() == 0:
                    basename = os.path.basename(save_path)
                    ckpt_path = os.path.join(save_path, f"{basename}.pt")
                    self.save_checkpoint(ckpt_path, save_info, self.save_weights_only)
            
            # For normal situation
            else:
                if dist.get_rank() == 0:
                    self.save_checkpoint(save_path, save_info, self.save_weights_only)
            
    def reset_metrics(self, stage) -> None:
        """
        Reset metrics for given stage
        Args:
            stage: "train", "valid" or "test"
        """
        for metric in self.metrics[stage].values():
            metric.reset()
    
    def get_log_dict(self, stage: str) -> dict:
        """
        Get log dict for the stage
        Args:
            stage: "train", "valid" or "test"

        Returns:
            A dictionary of metrics for the stage. Keys are metric names and values are metric values

        """
        return {name: metric.compute() for name, metric in self.metrics[stage].items()}
    
    def log_info(self, info: dict) -> None:
        """
        Record metrics during training and testing
        Args:
            info: dict of metrics
        """
        if getattr(self, "logger", None) is not None and dist.get_rank() == 0:
            info["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            info["epoch"] = self.epoch
            self.logger.log_metrics(info, step=self.step)

    def init_optimizers(self):
        copy_optimizer_kwargs = copy.deepcopy(self.optimizer_kwargs)
        
        # No decay for layer norm and bias
        no_decay = ['LayerNorm.weight', 'bias']
        weight_decay = copy_optimizer_kwargs.pop("weight_decay")

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer_cls = eval(f"torch.optim.{copy_optimizer_kwargs.pop('class')}")
        self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                       lr=self.lr_scheduler_kwargs['init_lr'],
                                       **copy_optimizer_kwargs)

        tmp_kwargs = copy.deepcopy(self.lr_scheduler_kwargs)
        lr_scheduler = tmp_kwargs.pop("class")
        self.lr_scheduler = eval(lr_scheduler)(self.optimizer, **tmp_kwargs)
    
    def configure_optimizers(self):
        return {"optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": self.lr_scheduler,
                                 "interval": "step",
                                 "frequency": 1}
                }
