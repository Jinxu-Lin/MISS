import argparse
import numpy as np
import random
import os
from typing import Iterable
from tqdm import tqdm


import torch
from torch import optim
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.func import functional_call, vmap, grad 
import torchvision
from trak.projectors import ProjectionType, CudaProjector

from Tools.Data import cifar2, cifar10, imagenet
from Tools.Models.resnet import resnet18, resnet34
from Tools.Models.resnet9 import resnet9

from trak.traker import TRAKer


dataset_num_classes = {
    'cifar2': 2,
    'cifar10': 10,
    'imagenet': 1000
}

dataset_len = {
    'cifar2': 10000,
    'cifar10': 50000,
    'imagenet': 1281167
}

testset_len = {
    'cifar2': 2000,
    'cifar10': 10000,
    'imagenet': 50000
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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


def get_out_to_loss_grad(
        model, weights, buffers, inputs, labels
    ) -> Tensor:
        logits = torch.func.functional_call(model, (weights, buffers), inputs)
        # here we are directly implementing the gradient instead of relying on autodiff to do
        # that for us
        ps = torch.nn.Softmax(-1)(logits)[
            torch.arange(logits.size(0)), labels
        ]
        return (1 - ps).clone().detach().unsqueeze(-1)


def parseArgs():

    parser = argparse.ArgumentParser(
        description="Image Classification Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        dest="seed", help='random seed')
    
    # Dataset
    parser.add_argument("--dataset", type=str, default='cifar2',
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--load-dataset", action="store_true", default=False,
                        dest="load_dataset", help='load local dataset')
    parser.add_argument("--dataset-dir", type=str, default=None,
                        dest="dataset_dir", help='dataset directory')
    parser.add_argument("--dataset-split", type=str, default='train',
                        dest="dataset_split", help='dataset split')
    parser.add_argument("--train-index-path", type=str, default=None,
                        dest="train_index_path", help='train index path')
    parser.add_argument("--test-index-path", type=str, default=None,
                        dest="test_index_path", help='test index path')
    parser.add_argument("--data-aug", action="store_true", default=True,
                        dest="data_aug", help='data augmentation')
    parser.add_argument("--resolution", type=int, default=32,
                        dest="resolution", help='resolution of the dataset')
    parser.add_argument("--center-crop", action="store_true", default=False,
                        dest="center_crop", help='center crop the dataset')
    parser.add_argument("--random-flip", action="store_true", default=False,
                        dest="random_flip", help='random flip the dataset')
    parser.add_argument("--batch-size", type=int, default=16, 
                        dest="batch_size", help="Batch size (per device) for the dataloader.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0,
                        dest="dataloader_num_workers", help="The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    
    # Model
    parser.add_argument("--model", type=str, default='resnet9',
                        dest="model", help='model to train on')
    parser.add_argument("--model-dir", type=str, default='./saved/models',
                        dest="model_dir", help='model directory')
    parser.add_argument("--model-name", type=str, default='model_10.pth',
                        dest="model_name", help='model name')
    
    # Save Dir
    parser.add_argument("--save-dir", type=str, default='./saved/grad',
                        dest="save_dir", help='save directory')
    
    return parser.parse_args()


def main(args):
    # seed
    set_seed(args.seed)

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset_split == 'train':
        loader = dataset_loader[args.dataset].get_train_loader(
            args,
        )
    else:
        loader = dataset_loader[args.dataset].get_test_loader(
            args,
        )

    # load model
    num_classes = dataset_num_classes[args.dataset]
    model = models[args.model](num_classes=num_classes)
    model_path = os.path.join(args.model_dir, args.model_name)
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()

    # get params and buffers
    func_weights = dict(model.named_parameters())
    func_buffers = dict(model.named_buffers())

    # normalize factor
    normalize_factor = torch.sqrt(
        torch.tensor(count_parameters(model), dtype=torch.float32)
    )

    # initialize save np array
    if args.dataset_split == 'train':
        dataset_size = dataset_len[args.dataset]
    else:
        dataset_size = testset_len[args.dataset]
    filename = os.path.join(args.save_dir, f'error.npy')
    loss_grads_store = np.memmap(filename, 
                             dtype=np.float32, 
                             mode='w+', 
                             shape=(dataset_size,1))

    index_start = 0

    for batch_idx, batch in enumerate(loader):
        print(f"{batch_idx}/{len(loader)}")

        index_end = index_start + len(batch["input"])

        inputs, labels = batch["input"].to(device), batch["label"].to(device)

        loss_grads = get_out_to_loss_grad(model, func_weights, func_buffers, inputs, labels)

        # save gradient
        loss_grads_store[index_start:index_end] = loss_grads.cpu().clone().detach().numpy()

        index_start = index_end


if __name__ == '__main__':
    args = parseArgs()
    main(args)