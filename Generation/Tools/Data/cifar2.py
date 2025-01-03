import os
import torch
import pickle
from datasets import load_dataset, load_from_disk
from torchvision import transforms


def get_train_loader(
        args,
):
    
    # load dataset
    if args.load_dataset:
        dataset = load_from_disk(
            os.path.join(args.dataset_dir, "train")
        )

    else:
        dataset = load_dataset(
            'cifar10',
            split="train"
        )
            
    # select CIFAR-2
    with open(args.train_index_path, 'rb') as handle:
        sub_idx = pickle.load(handle)
    dataset = dataset.select(sub_idx)

    # data augmentation
    augmentations = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )

    ])

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        labels = examples["label"]
        # Add label conversion here
        label_map = {3: 0, 5: 1}
        labels = [label_map[label] for label in labels]
        return {"input": images, "label": labels}


    if args.data_aug:
        dataset.set_transform(transform_images)

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )
    
    return train_dataloader


def get_test_loader(
        args,
):
    # load dataset
    if args.load_dataset:
        dataset = load_from_disk(
            os.path.join(args.dataset_dir, "test")
        )

    else:
        dataset = load_dataset(
            'cifar10',
            split="test",
            
        )

    # select CIFAR-2
    with open(args.test_index_path, 'rb') as handle:
        sub_idx = pickle.load(handle)
    dataset = dataset.select(sub_idx)

    # data augmentation
    augmentations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4942, 0.4851, 0.4504],
            std=[0.2467, 0.2429, 0.2616]
        )
    ])

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        labels = examples["label"]
        # Add label conversion here
        label_map = {3: 0, 5: 1}
        labels = [label_map[label] for label in labels]
        return {"input": images, "label": labels}

    if args.data_aug:
        dataset.set_transform(transform_images)

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )

    return test_dataloader