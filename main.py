import argparse
import os
from pathlib import Path
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from src.model import OpenSetModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from src.data.dataloader import prepare_data
from src.i3d import InceptionI3d
from src.utils.checkpointer import Checkpointer
from src.utils.metric_callback import MetricCallback
from src.utils.confusion_matrix import ConfusionMatrix
from src.utils.misc import (
    sample_estimator,
    compute_mahalanobis_score,
    k_means,
    compute_class_means_distance,
    extract_features,
    compute_thresholds,
    eval_source_only,
    test,
    cpu_benchmark
)
from collections import defaultdict
from torch.utils.data import DataLoader
from src.data.dataset import VideoDataset


def purge_classifier(state):
    del state["logits.conv3d.weight"]
    del state["logits.conv3d.bias"]
    return state


def prepare_model(args):
    file_path = Path(os.path.realpath(__file__)).parent

    if args.backbone == "i3d":
        rgb_model = InceptionI3d(in_channels=3)

        if args.backbone_pretrain == "kinetics":
            rgb_model.load_state_dict(
                purge_classifier(
                    torch.load(
                        file_path / "pretrained_models/i3d_rgb_imagenet+kinetics.pt"
                    )
                )
            )
    else:
        raise ValueError("Backbone not recognized!")

    model = OpenSetModel(rgb_model, **args.__dict__)

    return model


def find_number_of_classes(txt_file):
    labels = []
    with open(txt_file, "r") as txtfile:
        for line in txtfile:
            split_line = line.split()
            label = split_line[-1]
            labels.append(int(label))
    return len(list(set(labels)))


def count_per_class(txt_file):
    res = defaultdict(int)
    with open(txt_file, "r") as rgbfile:
        line_count = 0
        for line in rgbfile:
            if line_count == 0:
                line_count += 1
                continue
            else:
                label = line.split()[-1]
                res[label] += 1
                line_count += 1
    return res


def validate_args(args):
    if args.pretrained_model is not None:
        assert args.stage == 2, "Trying to load pretrained weights in stage 1"
        assert (
            not args.source_only
        ), "Trying to load pretrained weights for source only!"
    if args.open_set:
        assert not args.source_only, "Trying to do source only open set"
        assert (
            not args.remove_target_private
        ), "Removing target private samples for open set!"
    if args.adversarial_loss_weight:
        assert not args.source_only, "Using adversarial loss with source only!"
        if not args.enable_source_loss:
            print("Warning: source-only loss disabled!")
        assert args.stage == 2, "Using adversarial loss in stage 1!"
    if args.simclr_da_loss_weight:
        assert not args.source_only, "Using simclr DA with source only!"
        if not args.enable_source_loss:
            print("Warning: source-only loss disabled!")
        assert args.stage == 2, "Using simclr DA in stage 1!"
        if args.simclr_loss_weight_source or args.simclr_loss_weight_target:
            print("Warning: using double simclr!")
    if args.stage == 2:
        if not args.pretrained_model:
            print("Warning: running stage 2 without pretrained model!")
    if args.source_only:
        assert (
            args.remove_target_private
        ), "Running source only with target private classes!"
    if args.rejection_protocol == "simclr":
        assert (
            args.rejection_metric == "similarity"
        ), "Only similarity metric can be used with SimCLR!"
    if args.simclr_loss_weight_target:
        if args.stage == 1:
            print("Warning: using target for simclr in stage 1!")

    if args.aug_based_simclr_source:
        assert len(args.source_augmentations), "No source augmentation provided!"

    if args.aug_based_simclr_target:
        assert len(args.target_augmentations), "No target augmentation provided!"


def parse_args():
    SUP_OPT = ["sgd", "adam"]
    SUP_SCHED = ["reduce", "cosine", "step", "exponential", "warmup", "none"]
    SUP_AGGR = ["avg", "mlp", "mlp_weights", "lstm", "lstm_weights"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", type=str)
    parser.add_argument("--target_dataset", type=str)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--source_val_dataset", type=str, default=None)

    # optical flow datasets
    parser.add_argument("--source_dataset_of", type=str, default=None)
    parser.add_argument("--target_dataset_of", type=str, default=None)
    parser.add_argument("--val_dataset_of", type=str, default=None)

    # optimizer
    parser.add_argument("--optimizer", default="sgd", choices=SUP_OPT)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--multiple_lr", action="store_true")
    parser.add_argument("--lr_backbone", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    # scheduler
    parser.add_argument("--scheduler", choices=SUP_SCHED, default="reduce")
    parser.add_argument("--lr_steps", type=int, nargs="+")
    parser.add_argument("--num_restarts", type=int, default=2)

    # general settings
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epic_kitchens", action="store_true")
    parser.add_argument("--selection_factor", type=int, default=6)
    parser.add_argument("--target_ce_loss_weight", type=float, default=0.0)
    parser.add_argument("--no_sanity_checks", action="store_true")

    # training settings
    parser.add_argument("--resume_training_from", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--source_only", action="store_true")
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--pretrained_cop_model", type=str)
    parser.add_argument("--n_cop_clips", type=int, default=3)
    parser.add_argument("--cop_hidden_dim", type=int, default=512)
    parser.add_argument("--cop_dropout_p", type=float, default=0.0)
    parser.add_argument("--remove_target_private", action="store_true")
    parser.add_argument(
        "--backbone",
        type=str,
        default="i3d",
        choices=["i3d"],
    )
    parser.add_argument(
        "--backbone_pretrain",
        type=str,
        default="kinetics",
        choices=["kinetics", "none"],
    )
    parser.add_argument("--enable_classification_loss_at_epoch", type=int, default=0)
    parser.add_argument("--use_deep_classifier", action="store_true")
    parser.add_argument("--domain_classifier_hidden_size", type=int, default=100)
    parser.add_argument("--alpha", type=int, default=10)
    parser.add_argument("--project_feats", action="store_true")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--reverse_grad_from_epoch", type=int, default=0)
    parser.add_argument("--mlp_aggregator_n_layers", type=int, default=1)
    parser.add_argument("--mlp_aggregator_add_bn", action="store_true")
    parser.add_argument("--mlp_aggregator_dropout", type=float, default=0.0)
    parser.add_argument("--freeze_backbone", action="store_true")

    # loss settings
    parser.add_argument("--classification_loss_weight", type=float, default=1.0)
    parser.add_argument("--cop_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--cop_loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "pos_wise"],
    )
    parser.add_argument("--adversarial_loss_weight", type=float, default=0.0)
    parser.add_argument("--simclr_loss_weight_source", type=float, default=0.0)
    parser.add_argument("--simclr_loss_weight_target", type=float, default=0.0)
    parser.add_argument("--triplet_loss_weight_source", type=float, default=0.0)
    parser.add_argument("--triplet_loss_weight_target", type=float, default=0.0)
    parser.add_argument("--cop_loss_weight_source", type=float, default=0.0)
    parser.add_argument("--cop_loss_weight_target", type=float, default=0.0)
    parser.add_argument("--triplet_loss_margin", type=float, default=1.0)
    parser.add_argument("--fixed_perm", action="store_true")
    parser.add_argument("--simclr_proj_use_bn", action="store_true")
    parser.add_argument("--simclr_proj_n_layers", type=int, default=2)
    parser.add_argument("--simclr_da_loss_weight", type=float, default=0.0)

    # open set stuff
    parser.add_argument("--open_set", action="store_true")
    parser.add_argument(
        "--rejection_metric",
        type=str,
        choices=["entropy", "similarity"],
        default="entropy",
    )
    parser.add_argument("--open_set_threshold", type=float, default=0.5)
    parser.add_argument("--disable_rejection", action="store_true")
    parser.add_argument(
        "--rejection_protocol",
        type=str,
        default="cop",
        choices=[
            "cop",
            "entropy",
            "gt",
            "simclr+cmeans+cosine",
        ],
    )
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--enable_source_loss", action="store_true")
    parser.add_argument("--n_steps", type=int, default=3000)
    parser.add_argument(
        "--rejection_time", type=str, default="online", choices=["online", "offline"]
    )
    parser.add_argument("--queue_size", type=int, default=600)
    parser.add_argument("--proj_hidden_dim", type=int, default=512)
    parser.add_argument("--proj_output_dim", type=int, default=512)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument(
        "--cmeans_time", type=str, default="offline", choices=["offline", "online"]
    )

    # data stuff
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--n_clips", type=int, default=4)
    parser.add_argument("--source_augmentations", default=[], nargs="+")
    parser.add_argument("--target_augmentations", default=[], nargs="+")
    parser.add_argument("--val_augmentations", default=[], nargs="+")
    parser.add_argument("--merged", action="store_true")
    parser.add_argument("--clip_aggregation", type=str, default="avg", choices=SUP_AGGR)
    parser.add_argument("--aug_based_simclr_source", action="store_true")
    parser.add_argument("--aug_based_simclr_target", action="store_true")
    parser.add_argument("--use_extracted_feats", action="store_true")

    # wandb
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--cm", type=str, default="absolute", choices=["absolute", "percentage"]
    )

    # checkpoint dir
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint_frequency", type=int, default=1)

    # seed
    parser.add_argument("--seed", type=int, default=5)

    args = parser.parse_args()

    # find number of classes
    args.n_classes = find_number_of_classes(args.source_dataset)
    print("{} known classes".format(args.n_classes))

    args.items_per_class = count_per_class(args.val_dataset)

    return args


def main():

    args = parse_args()
    validate_args(args)

    seed_everything(args.seed)

    train_loader, val_loader, source_dataset, val_dataset = prepare_data(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        val_dataset=args.val_dataset,
        n_frames=args.n_frames,
        n_clips=args.n_clips,
        frame_size=args.frame_size,
        source_augmentations=args.source_augmentations,
        target_augmentations=args.target_augmentations,
        val_augmentations=args.val_augmentations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epic_kitchens=args.epic_kitchens,
        remove_target_private=args.remove_target_private,
        n_classes=args.n_classes,
        load_augmentations_source=(
            args.aug_based_simclr_source
            or args.triplet_loss_weight_source
            or args.simclr_da_loss_weight
        ),
        load_augmentations_target=(
            args.aug_based_simclr_target
            or args.triplet_loss_weight_target
            or args.simclr_da_loss_weight
        ),
        use_extracted_feats=args.use_extracted_feats,
        backbone=args.backbone,
    )

    args.target_num = len(train_loader.dataset.target_dataset)
    args.source_num = len(train_loader.dataset.source_dataset)
    args.val_num = len(val_loader.dataset)
    print("Source num {}".format(args.source_num))
    if not args.source_only:
        print("Target num: {}".format(args.target_num))
    print("Val num: {}".format(args.val_num))

    if args.rejection_protocol == "simclr+cmeans+cosine":
        if args.cmeans_time == "online":
            args.cmeans_train_dataset = source_dataset
    args.val_dataset = val_dataset

    model = prepare_model(args)

    if args.pretrained_model is not None:
        source_params = torch.load(args.pretrained_model, map_location="cpu")[
            "state_dict"
        ]
        model.load_state_dict(source_params, strict=False)

    callbacks = []
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        ckpt = Checkpointer(
            args,
            logdir=args.checkpoint_dir,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)
        cm = ConfusionMatrix(args)
        callbacks.append(cm)
        metric_callback = MetricCallback(n_classes=args.n_classes, filename=args.name)
        callbacks.append(metric_callback)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp",
        num_sanity_val_steps=0 if args.no_sanity_checks else 2,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
