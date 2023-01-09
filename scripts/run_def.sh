#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate exp

python3 ../main.py  \
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
  --gpus 0 1 2 3 \
  --num_workers 4 \
  --n_clips 3 \
  --name hmdb_ucf_stage1 \
  --project coloseo \
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

python3 ../main.py  \
  --source_dataset ../txt/epic-kitchens/D1_train_source.txt \
  --target_dataset ../txt/epic-kitchens/D2_train_target.txt \
  --val_dataset ../txt/epic-kitchens/D2_test_target.txt \
  --max_epochs 50 \
  --batch_size 4 \
  --optimizer sgd \
  --precision 16 \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --gpus 0 1 2 3 \
  --num_workers 4 \
  --n_clips 3 \
  --name ek12_stage1 \
  --project coloseo \
  --classification_loss_weight 1.0 \
  --simclr_loss_weight_source 1.0 \
  --simclr_loss_weight_target 1.0 \
  --triplet_loss_weight_source 0.1 \
  --triplet_loss_weight_target 0.1 \
  --aug_based_simclr_target \
  --target_augmentations horizontal color \
  --no_sanity_checks \
  --remove_target_private \
  --epic_kitchens \
  --stage 1 \
  --alpha 2 \
  --multiple_lr \
  --lr_backbone 0.0001 \
  --clip_aggregation mlp \
  --wandb


python3 ../main.py  \
  --source_dataset ../txt/hmdb_ucf/hmdb_train_source.txt \
  --target_dataset ../txt/hmdb_ucf/ucf_train_target.txt \
  --val_dataset ../txt/hmdb_ucf/ucf_test_target.txt \
  --max_epochs 50 \
  --batch_size 4 \
  --optimizer sgd \
  --precision 16 \
  --lr 0.01 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --gpus 0 1 2 3 \
  --num_workers 4 \
  --n_clips 3 \
  --name hmdb_ucf_stage2 \
  --project coloseo \
  --classification_loss_weight 1.0 \
  --triplet_loss_weight_source 0.1 \
  --triplet_loss_weight_target 0.1 \
  --simclr_da_loss_weight 1.0 \
  --no_sanity_checks \
  --source_augmentations horizontal color \
  --target_augmentations horizontal color \
  --open_set \
  --rejection_metric similarity \
  --rejection_protocol simclr+cmeans+cosine \
  --open_set_threshold 0.4 \
  --stage 2 \
  --multiple_lr \
  --lr_backbone 0.001 \
  --clip_aggregation mlp \
  --pretrained_model ../pretrained_models/hmdb_ucf_stage1.pt \
  --rejection_time online \
  --cmeans_time online \
  --k 6 \
  --wandb

python3 ../main.py  \
  --source_dataset ../txt/epic-kitchens-clean/D1_train_source.txt \
  --target_dataset ../txt/epic-kitchens-clean/D2_train_target.txt \
  --val_dataset ../txt/epic-kitchens-clean/D2_test_target.txt \
  --max_epochs 50 \
  --batch_size 2 \
  --optimizer sgd \
  --precision 16 \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --gpus 0 1 2 3 \
  --num_workers 2 \
  --n_clips 3 \
  --name ek12_stage2 \
  --project coloseo \
  --classification_loss_weight 1.0 \
  --simclr_da_loss_weight 1.0 \
  --triplet_loss_weight_source 0.1 \
  --triplet_loss_weight_target 0.1 \
  --open_set \
  --target_augmentations horizontal color \
  --source_augmentations horizontal color \
  --open_set_threshold 0.4 \
  --no_sanity_checks \
  --rejection_metric similarity \
  --rejection_protocol simclr+cmeans+cosine \
  --epic_kitchens \
  --stage 2 \
  --alpha 2 \
  --multiple_lr \
  --lr_backbone 0.0001 \
  --pretrained_model ../pretrained_models/ek12_stage1.pt \
  --rejection_time online \
  --cmeans_time online \
  --clip_aggregation mlp \
  --k 8 \
  --wandb

conda deactivate