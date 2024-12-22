import argparse
import numpy as np
import random
import os

import torch
from torch import optim

from trak.projectors import ProjectionType, AbstractProjector, CudaProjector

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    parser.add_argument("--model", type=str, default='resnet9',
                        dest="model", help='model to train on')
    parser.add_argument("--model-name", type=str, default='model_10.pth',
                        dest="model_name", help='model name')
    
    # Projector
    parser.add_argument("--dim", type=int, default=4096,
                        dest="dim", help='dimension of the projector')
    
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
    model.load_state_dict(f'./models/{args.model_name}')

    # initialize projector
    projector = CudaProjector(
        grad_dim=count_parameters(model), 
        proj_dim=args.dim,
        seed=42, 
        proj_type=ProjectionType.normal,
        device='cuda:0'
    )
    
    # model evaluation for gradient computation
    model.to(device)
    model.eval()

    for batch_idx, batch in enumerate(train_loader):
        print("batch_idx: ", batch_idx)
        data, labels = batch["input"].to(device), batch["label"].to(device)



if __name__ == "__main__":
    args = parseArgs()
    main(args)
