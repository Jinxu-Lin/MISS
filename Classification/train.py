import argparse
import numpy as np
import random
import os

import torch
from torch import optim

from Tools.Data import cifar2, cifar10, imagenet
from Tools.Models.resnet import resnet18, resnet34
from Tools.Models.resnet9 import resnet9


dataset_num_classes = {
    'cifar2': 2,
    'cifar10': 10,
    'imagenet': 1000
}

dataset_loader = {
    'cifar2': cifar2,
    'cifar10': cifar10,
    'imagenet': imagenet
}

models = {
    'resnet9': resnet9,
    'resnet18': resnet18,
}

def parseArgs():

    parser = argparse.ArgumentParser(
        description="Image Classification Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Dataset
    parser.add_argument("--dataset", type=str, default='cifar2',
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--load-dataset", action="store_true", default=False,
                        dest="load_dataset", help='load local dataset')
    parser.add_argument("--dataset-dir", type=str, default=None,
                        dest="dataset_dir", help='dataset directory')
    parser.add_argument("--train-index-path", type=str, default=None,
                        dest="train_index_path", help='index path')
    parser.add_argument("--test-index-path", type=str, default=None,
                        dest="test_index_path", help='index path')
    parser.add_argument("--data-aug", action="store_true", default=True,
                        dest="data_aug", help='data augmentation')
    parser.add_argument("--resolution", type=int, default=32,
                        dest="resolution", help='resolution of the dataset')
    parser.add_argument("--center-crop", action="store_true", default=False,
                        dest="center_crop", help='center crop the dataset')
    parser.add_argument("--random-flip", action="store_true", default=False,
                        dest="random_flip", help='random flip the dataset')
    parser.add_argument("--train-batch-size", type=int, default=64, 
                        dest="train_batch_size", help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--test-batch-size", type=int, default=256, 
                        dest="test_batch_size", help="Batch size (per device) for the test dataloader.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0,
                        dest="dataloader_num_workers", help="The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    
    # Model
    parser.add_argument("--model", type=str, default="resnet9",
                        dest="model", help="Model to use")

    # Optimiser
    parser.add_argument("--optimiser", type=str, default="sgd",
                        dest="optimiser", help="Optimiser to use")
    parser.add_argument("--learning-rate", type=float, default=0.4,
                        dest="learning_rate", help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        dest="momentum", help="Momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        dest="weight_decay", help="Weight decay")
    parser.add_argument("--nesterov", action="store_true", default=False,
                        dest="nesterov", help="Nesterov momentum")
    
    # Scheduler
    parser.add_argument("--scheduler", type=str, default="multi_step",
                        dest="scheduler", help="Scheduler to use")
    parser.add_argument("--lr-peak-epoch", type=int, default=5,
                        dest="lr_peak_epoch", help="Learning rate peak epoch")
    
    # Train
    parser.add_argument("--epochs", type=int, default=24,
                        dest="epochs", help="Number of epochs to train for")

    return parser.parse_args()

def main(args):

    # seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # load dataset
    train_loader = dataset_loader[args.dataset].get_train_loader(
        args,
    )
    test_loader = dataset_loader[args.dataset].get_test_loader(
        args,
    )

    # load model
    num_classes = dataset_num_classes[args.dataset]
    model = models[args.model](num_classes=num_classes)

    # load optimizer
    if args.optimiser == "sgd":
        opt_params = model.parameters()
        optimizer = optim.SGD(opt_params,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optimiser == "adam":
        opt_params = model.parameters()
        optimizer = optim.Adam(opt_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
        
    # load scheduler
    if args.scheduler == "multi_step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    elif args.scheduler == "Cyclic":
        def triangular_lr(epoch_iter):
            iters_per_epoch = len(train_loader)
            total_iters = args.epochs * iters_per_epoch
            peak_iter = args.lr_peak_epoch * iters_per_epoch
            # Triangular learning rate
            return np.interp(epoch_iter, [0, peak_iter, total_iters], [args.learning_rate, args.learning_rate * 10, args.learning_rate])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=triangular_lr)

    # train
    model = model.to(device)
    model.train()

    # create models dir
    os.makedirs("./models", exist_ok=True)

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            print("epoch: ", epoch, "batch_idx: ", batch_idx)
            data, labels = batch["input"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # save model to ./models
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./models/model_{epoch}.pth")
    # save model
    torch.save(model.state_dict(), f"./models/model_{args.epochs}.pth")


if __name__ == "__main__":
    args = parseArgs()
    main(args)
