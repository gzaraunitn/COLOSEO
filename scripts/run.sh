#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate exp

CUDA_VISIBLE_DEVICES=0,1 python3 ../main.py  \
  --source_dataset ../txt/hmdb_ucf/hmdb_train_source.txt \
  --target_dataset ../txt/hmdb_ucf/ucf_train_target.txt \
  --val_dataset ../txt/hmdb_ucf/ucf_test_target.txt \
  --max_epochs 50 \
  --batch_size 2 \
  --optimizer sgd \
  --precision 16 \
  --lr 0.01 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --gpus 0 1 \
  --num_workers 4 \
  --n_clips 3 \
  --name [3]_hmdb_ucf_stage1 \
  --project debug \
  --classification_loss_weight 1.0 \
  --triplet_loss_weight_source 0.1 \
  --triplet_loss_weight_target 0.1 \
  --simclr_loss_weight_source 1.0 \
  --simclr_loss_weight_target 1.0 \
  --no_sanity_checks \
  --remove_target_private \
  --aug_based_simclr_target \
  --source_augmentations horizontal color \
  --target_augmentations horizontal color \
  --stage 1 \
  --alpha 2 \
  --multiple_lr \
  --lr_backbone 0.001 \
  --clip_aggregation mlp \
  --wandb



#CUDA_VISIBLE_DEVICES=0,1 python3 ../main.py  \
#  --source_dataset ../txt/hmdb_ucf/hmdb_train_source.txt \
#  --target_dataset ../txt/hmdb_ucf/ucf_train_target.txt \
#  --val_dataset ../txt/hmdb_ucf/ucf_test_target.txt \
#  --max_epochs 50 \
#  --batch_size 2 \
#  --optimizer sgd \
#  --precision 16 \
#  --lr 0.01 \
#  --weight_decay 1e-4 \
#  --scheduler cosine \
#  --gpus 0 1 \
#  --num_workers 4 \
#  --n_clips 3 \
#  --name prova \
#  --project debug \
#  --classification_loss_weight 1.0 \
#  --triplet_loss_weight_source 0.1 \
#  --triplet_loss_weight_target 0.1 \
#  --simclr_da_loss_weight 1.0 \
#  --no_sanity_checks \
#  --source_augmentations horizontal color \
#  --target_augmentations horizontal color \
#  --open_set \
#  --rejection_metric similarity \
#  --rejection_protocol simclr+cmeans+cosine \
#  --open_set_threshold 0.4 \
#  --stage 2 \
#  --multiple_lr \
#  --lr_backbone 0.001 \
#  --clip_aggregation mlp \
#  --pretrained_model ../pretrained_models/3_hmdb_ucf_chen_stage1.pt \
#  --rejection_time online \
#  --cmeans_time online \
#  --k 6 \
#  --wandb

conda deactivate