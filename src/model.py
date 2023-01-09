from typing import Any, Dict, List
import numpy as np
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from datetime import datetime
from .utils.misc import (
    accuracy_at_k,
    weighted_mean,
    compute_entropy,
    compute_auroc,
    k_means,
    compute_class_means_distance,
    get_rank,
    gather,
    permute_clips,
    compute_thresholds,
    compute_acc,
)
from .utils.modules import (
    AvgPoolAggregation,
    Classifier,
    MLPAggregation,
    MLPAggregationWeights,
    LSTMAggregation,
    LSTMAggregationWeights,
    SimCLRProjector,
    COPModule,
)
from .data.dataset import VideoDataset
from .utils.label_smoothing import smooth_cross_entropy_loss
from .utils.loss import simclr_loss_func
from .utils.scheduler import CosineAnnealingWarmupRestarts
from typing import Any, Dict, List, Tuple
from sklearn.metrics import accuracy_score
from time import time


# noinspection PyCallingNonCallable
class OpenSetModel(pl.LightningModule):
    def __init__(self, rgb_model: nn.Module, n_classes: int, **kwargs):
        super().__init__()

        self.n_classes = n_classes
        self.kwargs = kwargs
        self.rgb_model = rgb_model

        # infer feat dimensions
        if self.kwargs["backbone"] == "i3d":
            with torch.no_grad():
                x = torch.zeros((2, 3, 16, 224, 224))
                out = self.rgb_model(x)
            self.feat_dim = out.size(1)
            self.clip_dim = self.feat_dim
        else:
            if "18" in self.kwargs["backbone"]:
                self.feat_dim = 512
            else:
                self.feat_dim = 2048

        self.start_time = 0

        clip_aggregation = self.kwargs["clip_aggregation"]
        if clip_aggregation == "avg":
            self.clip_aggregator = AvgPoolAggregation()
        elif clip_aggregation == "mlp":
            self.clip_aggregator = MLPAggregation(
                feature_size=self.feat_dim,
                n_clips=self.kwargs["n_clips"],
                layers=self.kwargs["mlp_aggregator_n_layers"],
                add_bn=self.kwargs["mlp_aggregator_add_bn"],
                dropout=self.kwargs["mlp_aggregator_dropout"],
            )
        elif clip_aggregation == "lstm":
            self.clip_aggregator = LSTMAggregation(feature_size=self.feat_dim)
        elif clip_aggregation == "mlp_weights":
            self.clip_aggregator = MLPAggregationWeights(
                feature_size=self.feat_dim,
                n_clips=self.kwargs["n_clips"],
                layers=self.kwargs["mlp_aggregator_n_layers"],
                add_bn=self.kwargs["mlp_aggregator_add_bn"],
                dropout=self.kwargs["mlp_aggregator_dropout"],
            )
        else:
            self.clip_aggregator = LSTMAggregationWeights(feature_size=self.feat_dim)

        if self.kwargs["use_extracted_feats"]:
            self.frame_aggregator = AvgPoolAggregation()

        self.log_file_name = None
        self.separation_plot_dir = None

        if self.kwargs["project_feats"]:
            self.projector = nn.Linear(self.feat_dim, self.kwargs["proj_dim"])
            self.feat_dim = self.kwargs["proj_dim"]

        if self.kwargs["use_deep_classifier"]:
            self.C1 = Classifier(
                input_dim=self.feat_dim,
                n_classes=self.n_classes,
                hidden_dim=self.feat_dim,
            )

            self.C2 = Classifier(
                input_dim=self.feat_dim,
                n_classes=self.n_classes + 1,
                hidden_dim=self.feat_dim,
            )
        else:
            self.C1 = nn.Linear(self.feat_dim, self.n_classes)
            self.C2 = nn.Linear(self.feat_dim, self.n_classes + 1)

        self._num_training_steps = None

        if self.kwargs["rejection_metric"] == "entropy":
            self.rejection_metric = compute_entropy
        elif self.kwargs["rejection_metric"] == "similarity":
            self.rejection_metric = self.find_nn_sim

        self.source_only = self.kwargs["source_only"]

        if self.kwargs["open_set"]:
            self.correct_per_class = [0 for _ in range(self.n_classes + 1)]
            self.instances_per_class = [0 for _ in range(self.n_classes + 1)]
            self.known_preds = []
            self.unknown_preds = []
            self.open_set_preds = []
            self.open_set_labels = []
            self.accept_mask = None

        self.simclr_projector = SimCLRProjector(
            input_dim=self.feat_dim,
            proj_hidden_dim=self.kwargs["proj_hidden_dim"],
            proj_output_dim=self.kwargs["proj_output_dim"],
            add_bn=self.kwargs["simclr_proj_use_bn"],
            n_layers=self.kwargs["simclr_proj_n_layers"],
        )

        self.train_dataloader = None
        self.centroids = torch.zeros((self.kwargs["k"], self.feat_dim)).to(self.device)
        if (
            self.kwargs["triplet_loss_weight_source"]
            or self.kwargs["triplet_loss_weight_target"]
        ):
            self.triplet_loss = nn.TripletMarginLoss(
                margin=self.kwargs["triplet_loss_margin"]
            )

        if (
            self.kwargs["cop_loss_weight_source"]
            or self.kwargs["cop_loss_weight_target"]
        ):
            self.cop_module = COPModule(
                device=self.device,
                n_clips=self.kwargs["n_cop_clips"],
                in_dim=self.feat_dim,
                hidden_dim=self.kwargs["cop_hidden_dim"],
                dropout_p=self.kwargs["cop_dropout_p"],
            )

        self.thresholds = None

        # cf.apply_channels_last(self)

    def configure_optimizers(self):
        # select optimizer
        if self.kwargs["optimizer"] == "sgd":
            optimizer = torch.optim.SGD
            extra_optimizer_args = {"momentum": 0.9}
        else:
            optimizer = torch.optim.Adam
            extra_optimizer_args = {}

        if self.kwargs["multiple_lr"]:
            classifiers_params = (
                list(self.C1.parameters())
                + list(self.C2.parameters())
                + list(self.clip_aggregator.parameters())
                + list(self.simclr_projector.parameters())
            )
            if (
                self.kwargs["cop_loss_weight_source"]
                or self.kwargs["cop_loss_weight_target"]
            ):
                classifiers_params += list(self.cop_module.parameters())
            optimizer = optimizer(
                [
                    {"params": classifiers_params},
                    {
                        "params": self.rgb_model.parameters(),
                        "lr": self.kwargs["lr_backbone"],
                    },
                ],
                lr=self.kwargs["lr"],
                weight_decay=self.kwargs["weight_decay"],
                **extra_optimizer_args,
            )
            optimizers = [optimizer]
        elif "resnet" in self.kwargs["backbone"]:
            resnet_parameters = self.rgb_model.parameters()
            resnet_p_ids = [id(p) for p in resnet_parameters]
            filtered_parameters = [
                p for p in self.parameters() if id(p) not in resnet_p_ids
            ]
            for resnet_param in resnet_parameters:
                resnet_param.requires_grad = False
            optimizer = optimizer(
                filtered_parameters,
                lr=self.kwargs["lr"],
                weight_decay=self.kwargs["weight_decay"],
                **extra_optimizer_args,
            )
            optimizers = [optimizer]
        elif self.kwargs["freeze_backbone"]:
            trainable_params = (
                list(self.simclr_projector.parameters())
                + list(self.C1.parameters())
                + list(self.C2.parameters())
                + list(self.clip_aggregator.parameters())
            )
            optimizer = optimizer(
                trainable_params,
                lr=self.kwargs["lr"],
                weight_decay=self.kwargs["weight_decay"],
                **extra_optimizer_args,
            )
            optimizers = [optimizer]
            for param in self.rgb_model.parameters():
                param.requires_grad = False
        else:
            optimizer = optimizer(
                self.parameters(),
                lr=self.kwargs["lr"],
                weight_decay=self.kwargs["weight_decay"],
                **extra_optimizer_args,
            )

            optimizers = [optimizer]

        schedulers = [self.get_scheduler(opt) for opt in optimizers]

        if schedulers[0] is None:
            return optimizers

        return optimizers, schedulers

    def get_scheduler(self, optimizer):
        if self.kwargs["scheduler"] == "none":
            return None
        else:
            if self.kwargs["scheduler"] == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, self.kwargs["max_epochs"]
                )
            elif self.kwargs["scheduler"] == "reduce":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            elif self.kwargs["scheduler"] == "step":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, self.kwargs["lr_steps"]
                )
            elif self.kwargs["scheduler"] == "exponential":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, self.kwargs["weight_decay"]
                )
            elif self.kwargs["scheduler"] == "warmup":
                scheduler = CosineAnnealingWarmupRestarts(
                    warmup_epochs=5,
                    optimizer=optimizer,
                    T_0=int(
                        self.kwargs["max_epochs"] / (self.kwargs["num_restarts"] + 1)
                    ),
                    eta_min=self.kwargs["lr"] * 1e-3,
                )
        return scheduler

    def load_weights(self):
        if self.kwargs["use_deep_classifier"]:
            save_weight = self.C1.fc3.weight.data.clone()
            save_bias = self.C1.fc3.bias.data.clone()
            self.C2.fc3.weight.data[: self.n_classes] = save_weight
            self.C2.fc3.bias.data[:] = torch.min(save_bias) - 1.0
            self.C2.fc3.bias.data[: self.n_classes] = save_bias
        else:
            save_weight = self.C1.weight.data.clone()
            save_bias = self.C1.bias.data.clone()
            self.C2.weight.data[: self.n_classes] = save_weight
            self.C2.bias.data[:] = torch.min(save_bias) - 1.0
            self.C2.bias.data[: self.n_classes] = save_bias

    @torch.no_grad()
    def find_nn_sim(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds the nearest neighbor of a sample.
        Args:
            z (torch.Tensor): a batch of projected features.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """

        sim, _ = (z @ self.centroids.T).max(dim=1)
        return sim

    def forward(self, x):

        # consider clip as extra video in the batch
        if self.kwargs["use_extracted_feats"]:
            video_feats = self.frame_aggregator(x)
        else:
            if self.kwargs["backbone"] == "i3d":
                b, n_clips, *_ = x.size()
                x = rearrange(x, "b n f c h w -> (b n) f c h w")
                clip_feats = self.rgb_model(x)
                clip_feats = clip_feats.view(b, n_clips, -1)
                video_feats = self.clip_aggregator(clip_feats)
            elif self.kwargs["toy"]:
                x = x.mean(dim=1)
                x = self.rgb_model(x)
                video_feats = x.reshape(x.size(0), -1)
            else:
                frame_feats = self.rgb_model(x)
                video_feats = torch.mean(frame_feats, dim=1)
                clip_feats = None

        if self.kwargs["project_feats"]:
            video_feats = self.projector(video_feats)

        pred = self.C1(video_feats)

        out = {
            "feat": video_feats,
            "pred": pred,
        }

        if (
            self.kwargs["triplet_loss_weight_source"]
            or self.kwargs["triplet_loss_weight_target"]
        ):
            if self.kwargs["fixed_perm"]:
                permuted_clip_feats = clip_feats[:, torch.randperm(clip_feats.size(1))]
            else:
                permuted_clip_feats = permute_clips(clip_feats)
            permuted_video_feats = self.clip_aggregator(permuted_clip_feats)
            out.update({"permuted_feat": permuted_video_feats})

        # perform cop
        if (
            self.kwargs["cop_loss_weight_source"]
            or self.kwargs["cop_loss_weight_target"]
        ):
            cop, cop_label = self.cop_module(clip_feats)
            out.update({"cop": cop, "cop_label": cop_label})

        return out

    @property
    def num_training_steps(self) -> int:
        """Training steps per epoch inferred from datamodule and devices."""

        if self._num_training_steps is None:
            if self.trainer.train_dataloader is None:
                self.train_dataloader()

            dataset_size = getattr(self, "dali_epoch_size", None) or len(
                self.trainer.train_dataloader.dataset
            )

            dataset_size = self.trainer.limit_train_batches * dataset_size

            num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
            if self.trainer.tpu_cores:
                num_devices = max(num_devices, self.trainer.tpu_cores)

            effective_batch_size = (
                self.kwargs["batch_size"]
                * self.trainer.accumulate_grad_batches
                * num_devices
            )
            self._num_training_steps = dataset_size // effective_batch_size

        return self._num_training_steps

    def compute_accept_mask(self, pred, y_trg, feat):
        if self.kwargs["rejection_protocol"] == "gt":
            accept_mask = y_trg != self.n_classes
        else:
            if self.kwargs["rejection_protocol"] == "simclr+cmeans+cosine":
                feat = F.normalize(feat, dim=-1)
                metric = self.rejection_metric(feat)
                accept_mask = metric > self.kwargs["open_set_threshold"]
            else:
                pred = F.softmax(pred, dim=1)
                metric = self.rejection_metric(pred)
                accept_mask = metric < self.kwargs["open_set_threshold"]
        return accept_mask

    def training_step(self, batch, batch_idx):

        if self.source_only:
            if (
                self.kwargs["aug_based_simclr_source"]
                or self.kwargs["triplet_loss_weight_source"]
            ):
                (X_src_aug_1, X_src_aug_2, y_src) = batch
            else:
                (X_src, y_src) = batch
        else:
            if (
                self.kwargs["aug_based_simclr_source"]
                or self.kwargs["triplet_loss_weight_source"]
                or self.kwargs["simclr_da_loss_weight"]
            ) and (
                self.kwargs["aug_based_simclr_target"]
                or self.kwargs["triplet_loss_weight_target"]
                or self.kwargs["simclr_da_loss_weight"]
            ):
                (
                    source_index,
                    X_src_aug_1,
                    X_src_aug_2,
                    y_src,
                    target_index,
                    X_trg_aug_1,
                    X_trg_aug_2,
                    y_trg,
                ) = batch
            elif (
                self.kwargs["aug_based_simclr_source"]
                or self.kwargs["triplet_loss_weight_source"]
            ):
                (
                    source_index,
                    X_src_aug_1,
                    X_src_aug_2,
                    y_src,
                    target_index,
                    X_trg,
                    y_trg,
                ) = batch
            elif (
                self.kwargs["aug_based_simclr_target"]
                or self.kwargs["triplet_loss_weight_target"]
            ):
                (
                    source_index,
                    X_src,
                    y_src,
                    target_index,
                    X_trg_aug_1,
                    X_trg_aug_2,
                    y_trg,
                ) = batch
            else:
                (
                    source_index,
                    X_src,
                    y_src,
                    target_index,
                    X_trg,
                    y_trg,
                ) = batch

        # ======================= SOURCE ======================= #

        if not (
            self.kwargs["aug_based_simclr_source"]
            or self.kwargs["triplet_loss_weight_source"]
            or self.kwargs["simclr_da_loss_weight"]
        ):
            out_src = self(X_src)
        else:
            out_src = out_src_1 = self(X_src_aug_1)
            out_src_2 = self(X_src_aug_2)

        pred_source = out_src["pred"]
        feat_source = out_src["feat"]

        # compute accuracy
        if self.n_classes >= 5:
            acc1_src, acc5_src = accuracy_at_k(pred_source, y_src, top_k=(1, 5))
        else:
            (acc1_src,) = accuracy_at_k(pred_source, y_src, top_k=(1,))
            acc5_src = 0

        losses = {}

        # classification loss
        classification_loss = F.cross_entropy(pred_source, y_src)

        if self.kwargs["stage"] == 1 or self.kwargs["enable_source_loss"]:
            if self.kwargs["stage"] == 1:
                if self.current_epoch >= int(
                    self.kwargs["enable_classification_loss_at_epoch"]
                ):
                    losses.update(
                        {
                            "source_cross_entropy": classification_loss
                            * self.kwargs["classification_loss_weight"],
                        }
                    )
            else:
                losses.update(
                    {
                        "source_cross_entropy": classification_loss
                        * self.kwargs["classification_loss_weight"],
                    }
                )

        # define output
        output = {
            "loss": sum(losses.values()),
            "train_acc1_src": acc1_src,
            "train_acc5_src": acc5_src,
            **losses,
        }

        # simclr loss
        if self.kwargs["simclr_loss_weight_source"]:
            assert feat_source.size(0) >= 2
            if self.kwargs["aug_based_simclr_source"]:
                feat_source_1 = out_src_1["feat"]
                feat_source_2 = out_src_2["feat"]
                z1 = self.simclr_projector(feat_source_1)
                z2 = self.simclr_projector(feat_source_2)
                z = torch.cat((z1, z2))
                n = z1.size(0)
                rank = get_rank()
                indices = torch.arange(
                    n * rank, n * (rank + 1), device=z1.device
                ).repeat(2)
            else:
                z = self.simclr_projector(feat_source)
                indices = y_src
            simclr_loss = simclr_loss_func(z, indices)
            losses.update(
                {
                    "simclr_loss_source": simclr_loss
                    * self.kwargs["simclr_loss_weight_source"],
                }
            )
            output.update({"loss": sum(losses.values())})

        # triplet loss
        if self.kwargs["triplet_loss_weight_source"]:
            feat_source_1 = out_src_1["feat"]
            feat_source_2 = out_src_2["feat"]
            permuted_feat = out_src_1["permuted_feat"]
            anchor = feat_source_1
            positive = feat_source_2
            negative = permuted_feat
            triplet_loss = self.triplet_loss(anchor, positive, negative)
            losses.update(
                {
                    "triplet_loss_source": triplet_loss
                    * self.kwargs["triplet_loss_weight_source"]
                }
            )
            output.update({"loss": sum(losses.values())})

        if self.kwargs["cop_loss_weight_source"]:
            # cop loss
            cop = out_src["cop"]
            cop_labels = out_src["cop_label"]
            cop_loss = F.cross_entropy(cop, cop_labels)
            cop_acc, _ = accuracy_at_k(cop, cop_labels, top_k=(1, 5))

            # source loss
            losses.update(
                {
                    "cop_loss_source": cop_loss * self.kwargs["cop_loss_weight_source"],
                }
            )

            # define output
            output.update({"loss": sum(losses.values())})

        # ======================= TARGET ======================= #
        if not (
            self.kwargs["aug_based_simclr_target"]
            or self.kwargs["triplet_loss_weight_target"]
            or self.kwargs["simclr_da_loss_weight"]
        ):
            out_trg = self(X_trg)

        else:
            out_trg = out_trg_1 = self(X_trg_aug_1)
            out_trg_2 = self(X_trg_aug_2)

        pred_target = out_trg["pred"]
        feat_target = out_trg["feat"]

        # compute_accuracy
        if self.n_classes >= 5:
            acc1_trg, acc5_trg = accuracy_at_k(pred_target, y_trg, top_k=(1, 5))
        else:
            (acc1_trg,) = accuracy_at_k(pred_target, y_trg, top_k=(1,))
            acc5_trg = 0

        accepted_features = feat_target
        accept_mask = torch.ones(self.kwargs["batch_size"], device=self.device)

        # compute accepted/rejected
        if self.kwargs["open_set"]:

            assert self.kwargs["stage"] == 2, "Doing open set in stage 1!"

            accept_mask = self.compute_accept_mask(pred_target, y_trg, feat_target)
            if self.kwargs["rejection_time"] == "offline":
                if self.current_epoch == 0:
                    self.accept_mask[target_index[accept_mask]] = 1
                else:
                    accept_mask = self.accept_mask[target_index]
            rejected_features = feat_target[~accept_mask]
            accepted_features = feat_target[accept_mask]
            feats = torch.cat((feat_source, rejected_features))
            outs = self.C2(feats)
            fake_labels = (
                torch.ones(
                    rejected_features.size(0), device=self.device, dtype=torch.long
                )
                * self.n_classes
            )
            combined_labels = torch.cat((y_src, fake_labels))
            combined_cross_entropy = F.cross_entropy(outs, combined_labels)
            losses.update(
                {
                    "combined_cross_entropy": self.kwargs[
                        "classification_loss_weight"
                    ]
                    * combined_cross_entropy
                }
            )

            # compute_accuracy
            if self.n_classes >= 5:
                acc1_trg, acc5_trg = accuracy_at_k(
                    outs, combined_labels, top_k=(1, 5)
                )
            else:
                (acc1_trg,) = accuracy_at_k(outs, combined_labels, top_k=(1,))
                acc5_trg = 0

        # simclr loss
        if self.kwargs["simclr_loss_weight_target"]:
            assert feat_target.size(0) >= 2
            if self.kwargs["aug_based_simclr_target"]:
                feat_target_1 = out_trg_1["feat"]
                feat_target_2 = out_trg_2["feat"]
                z1 = self.simclr_projector(feat_target_1)
                z2 = self.simclr_projector(feat_target_2)
                z = torch.cat((z1, z2))
                n = z1.size(0)
                rank = get_rank()
                indices = torch.arange(
                    n * rank, n * (rank + 1), device=z1.device
                ).repeat(2)
                mask = accept_mask.repeat(2).to(z1.device)
            else:
                indices = pred_target.max(dim=1)[1]
                z = self.simclr_projector(feat_target)
                selection_target = (
                    compute_entropy(F.softmax(pred_target, dim=1))
                    < math.log(self.n_classes) / self.kwargs["selection_factor"]
                ).to(accept_mask.device)
                mask = torch.logical_and(accept_mask, selection_target)
            simclr_loss = simclr_loss_func(z, indices, mask)
            losses.update(
                {
                    "simclr_loss_target": simclr_loss
                    * self.kwargs["simclr_loss_weight_target"],
                }
            )
            output.update({"loss": sum(losses.values())})

        # triplet loss
        if self.kwargs["triplet_loss_weight_target"]:
            feat_target_1 = out_trg_1["feat"]
            feat_target_2 = out_trg_2["feat"]
            permuted_feat = out_trg_1["permuted_feat"]
            anchor = feat_target_1
            positive = feat_target_2
            negative = permuted_feat
            triplet_loss = self.triplet_loss(anchor, positive, negative)
            losses.update(
                {
                    "triplet_loss_source": triplet_loss
                    * self.kwargs["triplet_loss_weight_target"]
                }
            )
            output.update({"loss": sum(losses.values())})

        if self.kwargs["cop_loss_weight_target"]:
            # cop loss
            cop = out_trg["cop"]
            cop_labels = out_trg["cop_label"]
            cop_loss = F.cross_entropy(cop, cop_labels)
            cop_acc, _ = accuracy_at_k(cop, cop_labels, top_k=(1, 5))

            # source loss
            losses.update(
                {
                    "cop_loss_target": cop_loss
                    * self.kwargs["cop_loss_weight_target"],
                }
            )

            # define output
            output.update({"loss": sum(losses.values())})

        # cross-domain contrastive loss
        if self.kwargs["simclr_da_loss_weight"]:
            feat_source_1 = out_src_1["feat"]
            feat_source_2 = out_src_2["feat"]
            feat_target_1 = out_trg_1["feat"]
            feat_target_2 = out_trg_2["feat"]
            z_s_1 = self.simclr_projector(feat_source_1)
            z_s_2 = self.simclr_projector(feat_source_2)
            z_t_1 = self.simclr_projector(feat_target_1)
            z_t_2 = self.simclr_projector(feat_target_2)
            z = torch.cat((z_s_1, z_s_2, z_t_1, z_t_2))
            pseudo_y_trg = pred_target.max(dim=1)[1]
            indices = torch.cat((y_src, y_src, pseudo_y_trg, pseudo_y_trg))
            selection_target = (
                compute_entropy(F.softmax(pred_target, dim=1))
                < math.log(self.n_classes) / self.kwargs["selection_factor"]
            ).to(accept_mask.device)
            target_mask = torch.logical_and(accept_mask, selection_target)
            pseudo_label_acc = compute_acc(
                y_trg[target_mask], pseudo_y_trg[target_mask]
            )
            source_mask = torch.ones_like(target_mask)
            mask = torch.cat((source_mask, source_mask, target_mask, target_mask))
            simclr_da_loss = simclr_loss_func(z, indices, mask)
            losses.update(
                {
                    "simclr_da_loss": simclr_da_loss
                    * self.kwargs["simclr_da_loss_weight"],
                }
            )
            output.update(
                {"loss": sum(losses.values()), "pseudo_label_acc": pseudo_label_acc}
            )

        # update output
        output.update(
            {
                "loss": sum(losses.values()),
                "train_acc1_trg": acc1_trg,
                "train_acc5_trg": acc5_trg,
                **losses,
            }
        )

        self.log_dict(output, on_epoch=True, sync_dist=True)

        return output["loss"]

    def on_fit_start(self) -> None:
        if self.kwargs["stage"] == 2:
            self.load_weights()
        self.log_file_name = "logs/{}_{}".format(
            self.kwargs["name"], datetime.now().strftime("%m_%d_%Y_%H_%M_%S.txt")
        )
        with open(self.log_file_name, "w") as logfile:
            for key, value in self.kwargs.items():
                logfile.write("{} = {}\n".format(key, value))
            logfile.write("----------------------------------------\n")

        if self.kwargs["open_set"]:
            self.correct_per_class = [0 for _ in range(self.n_classes + 1)]
            self.instances_per_class = [0 for _ in range(self.n_classes + 1)]
            self.known_preds = []
            self.unknown_preds = []
            self.open_set_preds = []
            self.open_set_labels = []
            self.accept_mask = torch.zeros(self.kwargs["target_num"])
            self.separation_plot_dir = "separation_plots/{}_{}".format(
                self.kwargs["name"], datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            )

    def on_train_epoch_start(self):
        print("Dataset size: {}".format(len(self.trainer.train_dataloader.dataset)))
        if self.kwargs["open_set"]:
            self.correct_per_class = [0 for _ in range(self.n_classes + 1)]
            self.instances_per_class = [0 for _ in range(self.n_classes + 1)]
            self.known_preds = []
            self.unknown_preds = []
            self.open_set_preds = []
            self.open_set_labels = []
            if self.current_epoch == 1:
                self.accept_mask = self.accept_mask > 0
            if self.kwargs["rejection_protocol"] == "simclr+cmeans+cosine":
                if self.kwargs["cmeans_time"] == "online":
                    loader = DataLoader(
                        self.kwargs["cmeans_train_dataset"],
                        batch_size=1,
                        shuffle=True,
                        num_workers=self.kwargs["num_workers"],
                        pin_memory=True,
                        drop_last=True,
                    )
                    self.centroids = compute_class_means_distance(
                        model=self,
                        train_loader=loader,
                        test_loader=None,
                        aug_based=(
                            self.kwargs["aug_based_simclr_source"]
                            or self.kwargs["triplet_loss_weight_source"]
                            or self.kwargs["simclr_da_loss_weight"]
                        ),
                    )["means"]
                    self.centroids = gather(self.centroids.to(self.device))[
                        : self.n_classes
                    ]

        self.start_time = time()

    def validation_step(self, batch, batch_idx):
        X, target = batch
        batch_size = X.size(0)

        out = self(X)

        feat = out["feat"]

        if self.kwargs["open_set"]:
            pred = self.C2(feat)
        else:
            pred = out["pred"]

        if self.n_classes >= 5:
            val_acc1, val_acc5 = accuracy_at_k(pred, target, top_k=(1, 5))
        else:
            (val_acc1,) = accuracy_at_k(pred, target, top_k=(1,))
            val_acc5 = 0
        # val_loss = F.cross_entropy(pred, target)

        correct_per_class = []
        instances_per_class = []

        if self.kwargs["open_set"]:
            pred_label = pred.argmax(dim=1)
            for index in range(target.size(0)):
                label = target[index]
                predicted_label = pred_label[index]
                self.instances_per_class[label] += 1
                if label.item() == predicted_label.item():
                    self.correct_per_class[label] += 1

            # store prediction
            known_mask = target < self.n_classes
            known = pred[known_mask]
            unknown = pred[~known_mask].data.cpu().numpy()
            self.known_preds.append(known.data.cpu().numpy())
            self.unknown_preds.append(unknown)

            p = pred_label == self.n_classes
            l = target == self.n_classes
            self.open_set_preds.append(p.data.cpu().numpy())
            self.open_set_labels.append(l.data.cpu().numpy())

        results = {
            "batch_size": batch_size,
            "outputs": pred,
            "targets": target,
            "validation_accuracy": val_acc1,
            "validation_accuracy_at5": val_acc5,
            # "validation_loss": val_loss,
        }

        return results

    def on_train_epoch_end(self):
        current_time = time()
        epoch_time = current_time - self.start_time
        samples_per_second = (
            self.num_training_steps * self.kwargs["batch_size"]
        ) / epoch_time
        with open("samples_per_second.txt", "a") as logfile:
            logfile.write("{}: {}".format(self.current_epoch, samples_per_second))
        self.log_dict({"samples_per_second": samples_per_second}, sync_dist=True)

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):

        val_acc1 = weighted_mean(outs, "validation_accuracy", "batch_size")
        val_acc5 = weighted_mean(outs, "validation_accuracy_at5", "batch_size")
        # val_loss = weighted_mean(outs, "validation_loss", "batch_size")
        log = {
            # "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
        }

        if self.kwargs["open_set"]:

            correct_per_class = gather(
                torch.tensor(self.correct_per_class, device=self.device)
            )
            instances_per_class = gather(
                torch.tensor(self.instances_per_class, device=self.device)
            )

            accuracy_per_class = np.array(correct_per_class.cpu()) / np.array(
                instances_per_class.cpu()
            )
            closed_accuracy = accuracy_per_class[: self.n_classes].mean()
            print("Accuracy per class: {}".format(accuracy_per_class))
            print("Closed accuracy: {}".format(closed_accuracy))
            open_accuracy = accuracy_per_class[-1]
            h_score = (
                2 * closed_accuracy * open_accuracy / (closed_accuracy + open_accuracy)
            )
            known_preds = np.concatenate(self.known_preds, 0)
            unknown_preds = np.concatenate(self.unknown_preds, 0)
            x1, x2 = np.max(known_preds, axis=1), np.max(unknown_preds, axis=1)
            try:
                auroc_score = compute_auroc(x1, x2)
            except:
                auroc_score = None
                print("auroc error")

            open_set_preds = np.concatenate(self.open_set_preds, 0)
            open_set_labels = np.concatenate(self.open_set_labels, 0)
            binary_open_set_acc = accuracy_score(open_set_preds, open_set_labels)
            log.update(
                {
                    "closed_acc": closed_accuracy,
                    "open_acc": open_accuracy,
                    "h_score": h_score,
                    "binary_open_set_acc": binary_open_set_acc,
                }
            )

            if auroc_score is not None:
                log.update({"auroc_score": auroc_score})

        with open(self.log_file_name, "a") as logfile:
            logfile.write("Epoch {}\n".format(self.current_epoch))
            # logfile.write("    Validation loss: {}\n".format(val_loss))
            logfile.write("    Validation accuracy: {}\n".format(val_acc1))
            if self.kwargs["open_set"]:
                logfile.write("    Closed accuracy: {}\n".format(closed_accuracy))
                logfile.write("    Open accuracy: {}\n".format(open_accuracy))
                logfile.write("    H-score: {}\n".format(h_score))
                logfile.write("    AUROC score: {}\n".format(auroc_score))
                logfile.write(
                    "    Binary open set acc: {}\n".format(binary_open_set_acc)
                )
            logfile.write("----------------------------------------\n")

        self.log_dict(log, sync_dist=True)
