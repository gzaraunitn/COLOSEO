
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
import torch
from pytorch_lightning.metrics import ConfusionMatrix as confusion_matrix
from pytorch_lightning.callbacks import Callback
import numpy as np

# handles generation of confusion matrix
class ConfusionMatrix(Callback):
    def __init__(self, args):
        self.args = args

    def on_train_epoch_start(self, trainer, module):
        self.outputs_s = []
        self.targets_s = []

        self.outputs_t = []
        self.targets_t = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for output in outputs:
            if "out_s" in output:
                self.outputs_s.append(outputs["out_s"].cpu())
                self.targets_s.append(outputs["y_source"].cpu())

            if "out_t" in output:
                self.outputs_t.append(outputs["out_t"].cpu())
                self.targets_t.append(outputs["y_target"].cpu())

    def on_train_epoch_end(self, trainer, module, outputs):
        if trainer.is_global_zero:
            for name, outputs, targets in zip(
                ["source", "target"],
                [self.outputs_s, self.outputs_t],
                [self.targets_s, self.targets_t],
            ):
                if len(outputs):
                    outputs = torch.cat(outputs)
                    targets = torch.cat(targets)

                    preds = outputs.float().max(dim=1)[1]

                    cm = confusion_matrix(module.n_classes)(preds, targets).cpu()
                    sns.set(rc={"figure.figsize": (30, 30), "font.size": 15})
                    sns.set(font_scale=2)
                    ax = sns.heatmap(data=(cm.numpy()/np.sum(cm.numpy())), fmt='.2%', annot=True, cmap="OrRd")
                    values = list(range(cm.size(0)))
                    ax.set_xticklabels(values, rotation=45, fontsize="large")
                    ax.set_yticklabels(values, rotation=90, fontsize="large")
                    plt.tight_layout()
                    if self.args.wandb:
                        wandb.log({f"train_{name}_cm": wandb.Image(ax)}, commit=False)
                        plt.close()

    def on_validation_epoch_start(self, trainer, module):
        self.outputs = []
        self.targets = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.outputs.append(outputs["outputs"])
        self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, module):
        if trainer.is_global_zero:
            self.outputs = torch.cat(self.outputs)
            self.targets = torch.cat(self.targets)

            if not self.args.open_set:
                preds = self.outputs.float().max(dim=1)[1]
            else:
                preds = self.outputs
            targets = self.targets

            preds = preds.cpu()
            targets = targets.cpu()

            if self.args.open_set:
                n_classes = module.n_classes + 1
            else:
                n_classes = module.n_classes
            cm = confusion_matrix(n_classes)(preds, targets).cpu()
            if cm.size():
                sns.set(rc={"figure.figsize": (30, 30), "font.size": 15})
                sns.set(font_scale=2)
                if self.args.cm == "percentage":
                    ax = sns.heatmap(data=(cm.numpy()/np.sum(cm.numpy())), fmt='.2%', annot=True, cmap="OrRd")
                else:
                    ax = sns.heatmap(data=(cm.numpy()), annot=True, cmap="OrRd")
                values = list(range(cm.size(0)))
                ax.set_xticklabels(values, rotation=45, fontsize="large")
                ax.set_yticklabels(values, rotation=90, fontsize="large")
                plt.tight_layout()
                if self.args.wandb:
                    wandb.log({"val_cm": wandb.Image(ax)}, commit=False)
                    plt.close()
