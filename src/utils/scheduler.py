import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
):
    def __init__(self, warmup_epochs, *args, **kwargs):

        super(CosineAnnealingWarmupRestarts, self).__init__(*args, **kwargs)

        # Init optimizer with low learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.eta_min

        self.warmup_epochs = warmup_epochs

        # Get target LR after warmup is complete
        target_lr = (
            self.eta_min
            + (self.base_lrs[0] - self.eta_min)
            * (1 + math.cos(math.pi * warmup_epochs / self.T_i))
            / 2
        )

        # Linearly interpolate between minimum lr and target_lr
        linear_step = (target_lr - self.eta_min) / self.warmup_epochs
        self.warmup_lrs = [
            self.eta_min + linear_step * (n + 1) for n in range(warmup_epochs)
        ]

    def step(self, epoch=None):

        # Called on super class init
        if epoch is None:
            super(CosineAnnealingWarmupRestarts, self).step(epoch=epoch)

        else:
            if epoch < self.warmup_epochs:
                lr = self.warmup_lrs[epoch]
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # Fulfill misc super() funcs
                self.last_epoch = math.floor(epoch)
                self.T_cur = epoch
                self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

            else:

                super(CosineAnnealingWarmupRestarts, self).step(epoch=epoch)
