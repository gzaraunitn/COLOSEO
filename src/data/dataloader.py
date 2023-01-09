from torch.utils.data import DataLoader

from .dataset import VideoDataset, VideoDatasetContrastive, VideoDatasetSourceAndTarget


def prepare_datasets(
    source_dataset,
    target_dataset,
    val_dataset,
    source_augmentations=None,
    target_augmentations=None,
    val_augmentations=None,
    n_frames=4,
    n_clips=4,
    frame_size=224,
    epic_kitchens=False,
    remove_target_private=False,
    n_classes=12,
    aug_based_simclr_source=False,
    aug_based_simclr_target=False,
    use_extracted_feats=False,
    backbone="i3d",
):

    if aug_based_simclr_source:
        dataset_class_source = VideoDatasetContrastive
    else:
        dataset_class_source = VideoDataset

    if aug_based_simclr_target:
        dataset_class_target = VideoDatasetContrastive
    else:
        dataset_class_target = VideoDataset

    source_dataset = dataset_class_source(
        source_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        augmentations=source_augmentations,
        epic_kitchens=epic_kitchens,
        remove_target_private=remove_target_private,
        n_classes=n_classes,
        use_extracted_feats=use_extracted_feats,
        backbone=backbone,
    )
    target_dataset = dataset_class_target(
        target_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        augmentations=target_augmentations,
        epic_kitchens=epic_kitchens,
        n_classes=n_classes,
        use_extracted_feats=use_extracted_feats,
        backbone=backbone,
    )
    source_n_target_dataset = VideoDatasetSourceAndTarget(
        source_dataset, target_dataset
    )

    val_dataset = VideoDataset(
        val_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        epic_kitchens=epic_kitchens,
        remove_target_private=remove_target_private,
        n_classes=n_classes,
        use_extracted_feats=use_extracted_feats,
        backbone=backbone,
        augmentations=val_augmentations,
    )

    return source_n_target_dataset, val_dataset, source_dataset, val_dataset


# prepares dataloaders given input datasets
def prepare_dataloaders(train_dataset, val_dataset, batch_size=64, num_workers=4):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# prepares datasets and dataloaders
def prepare_data(
    source_dataset,
    target_dataset,
    val_dataset,
    n_frames=16,
    n_clips=1,
    frame_size=224,
    source_augmentations=None,
    target_augmentations=None,
    val_augmentations=None,
    batch_size=64,
    num_workers=4,
    epic_kitchens=False,
    remove_target_private=False,
    n_classes=12,
    load_augmentations_source=False,
    load_augmentations_target=False,
    use_extracted_feats=False,
    backbone="i3d",
):

    (
        source_n_target_dataset,
        val_dataset,
        source_dataset,
        val_dataset,
    ) = prepare_datasets(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        val_dataset=val_dataset,
        source_augmentations=source_augmentations,
        target_augmentations=target_augmentations,
        val_augmentations=val_augmentations,
        n_frames=n_frames,
        n_clips=n_clips,
        frame_size=frame_size,
        epic_kitchens=epic_kitchens,
        remove_target_private=remove_target_private,
        n_classes=n_classes,
        aug_based_simclr_source=load_augmentations_source,
        aug_based_simclr_target=load_augmentations_target,
        use_extracted_feats=use_extracted_feats,
        backbone=backbone,
    )
    source_n_target_loader, val_loader = prepare_dataloaders(
        source_n_target_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return source_n_target_loader, val_loader, source_dataset, val_dataset


if __name__ == "__main__":
    data = VideoDataset(
        "/data/datasets/mixamo_datasets/mixamo38",
        frame_size=224,
        n_clips=4,
        augmentations=["color", "horizontal", "spatial", "gaussian", "gray"],
    )
    *x, y = data[1]
