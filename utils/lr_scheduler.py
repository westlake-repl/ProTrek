import math

from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class ConstantLRScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 last_epoch: int = -1,
                 verbose: bool = False,
                 init_lr: float = 0.,
                 ):
        """
        This is an implementation of constant learning rate scheduler.
        Args:
            optimizer: Optimizer

            last_epoch: The index of last epoch. Default: -1

            verbose: If ``True``, prints a message to stdout for each update. Default: ``False``

            init_lr: Initial learning rate
        """
        
        self.init_lr = init_lr
        super().__init__(optimizer, last_epoch, verbose)
    
    def state_dict(self):
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            raise RuntimeError(
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )
        
        return [self.init_lr for group in self.optimizer.param_groups]
    

class CosineAnnealingLRScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 last_epoch: int = -1,
                 verbose: bool = False,
                 init_lr: float = 0.,
                 max_lr: float = 4e-4,
                 final_lr: float = 4e-5,
                 warmup_steps: int = 2000,
                 cosine_steps: int = 10000,
                 ):
        """
        This is an implementation of cosine annealing learning rate scheduler.
        Args:
            optimizer: Optimizer
            
            last_epoch: The index of last epoch. Default: -1
            
            verbose: If ``True``, prints a message to stdout for each update. Default: ``False``
            
            init_lr: Initial learning rate
            
            max_lr: Maximum learning rate after warmup
            
            final_lr: Final learning rate after decay
            
            warmup_steps: Number of steps for warmup
            
            cosine_steps: Number of steps for cosine annealing
        """
        
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps
        super(CosineAnnealingLRScheduler, self).__init__(optimizer, last_epoch, verbose)
        
    def state_dict(self):
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            raise RuntimeError(
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )
        
        step_no = self.last_epoch
        
        if step_no <= self.warmup_steps:
            lr = self.init_lr + step_no / self.warmup_steps * (self.max_lr - self.init_lr)
        
        else:
            lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) \
                    * (1 + math.cos(math.pi * (step_no - self.warmup_steps) / self.cosine_steps))
        
        return [lr for group in self.optimizer.param_groups]


class Esm2LRScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 last_epoch: int = -1,
                 verbose: bool = False,
                 init_lr: float = 0.,
                 max_lr: float = 4e-4,
                 final_lr: float = 4e-5,
                 warmup_steps: int = 2000,
                 start_decay_after_n_steps: int = 500000,
                 end_decay_after_n_steps: int = 5000000,
                 on_use: bool = True,
                 ):
        """
        This is an implementation of ESM2's learning rate scheduler.
        Args:
            optimizer: Optimizer
            
            last_epoch: The index of last epoch. Default: -1
            
            verbose: If ``True``, prints a message to stdout for each update. Default: ``False``
            
            init_lr: Initial learning rate
            
            max_lr: Maximum learning rate after warmup
            
            final_lr: Final learning rate after decay
            
            warmup_steps: Number of steps for warmup
            
            start_decay_after_n_steps: Start decay after this number of steps
            
            end_decay_after_n_steps: End decay after this number of steps
            
            on_use: Whether to use this scheduler. If ``False``, the scheduler will not change the learning rate
            and will only use the ``init_lr``. Default: ``True``
        """
        
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.start_decay_after_n_steps = start_decay_after_n_steps
        self.end_decay_after_n_steps = end_decay_after_n_steps
        self.on_use = on_use
        super(Esm2LRScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def state_dict(self):
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            raise RuntimeError(
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )

        step_no = self.last_epoch
        if not self.on_use:
            return [base_lr for base_lr in self.base_lrs]

        if step_no <= self.warmup_steps:
            lr = self.init_lr + step_no / self.warmup_steps * (self.max_lr - self.init_lr)
        
        elif step_no <= self.start_decay_after_n_steps:
            lr = self.max_lr
        
        elif step_no <= self.end_decay_after_n_steps:
            portion = (step_no - self.start_decay_after_n_steps) / (self.end_decay_after_n_steps - self.start_decay_after_n_steps)
            lr = self.max_lr - portion * (self.max_lr - self.final_lr)
           
        else:
            lr = self.final_lr
    
        return [lr for group in self.optimizer.param_groups]