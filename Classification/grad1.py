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

from trak.projectors import ProjectionType, CudaProjector

from Tools.Data import cifar2, cifar10, imagenet
from Tools.Models.resnet import resnet18, resnet34
from Tools.Models.resnet9 import resnet9


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


def output_function(
        model: Module,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        image: Tensor,
        label: Tensor,
    ) -> Tensor:
        logits = torch.func.functional_call(model, (weights, buffers), image.unsqueeze(0))
        bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        # remove the logits of the correct labels from the sum
        # in logsumexp by setting to -torch.inf
        cloned_logits[bindex, label.unsqueeze(0)] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()


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
    
    # Projector
    parser.add_argument("--dim", type=int, default=4096,
                        dest="dim", help='dimension of the projector')
    
    # Save Dir
    parser.add_argument("--save-dir", type=str, default='./saved/grad',
                        dest="save_dir", help='save directory')
    
    return parser.parse_args()


def main(args):
    # seed
    set_seed(args.seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
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

    # initialize projector
    projector = CudaProjector(
        grad_dim=count_parameters(model), 
        proj_dim=args.dim,
        seed=args.seed, 
        proj_type=ProjectionType.normal,
        device=device,
        max_batch_size=16
    )

    # initialize save np array
    if args.dataset_split == 'train':
        filename = os.path.join('{}/train-{}.npy'.format(
            args.save_dir, args.dim
        ))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        dstore_keys = np.memmap(filename, 
            dtype=np.float32, 
            mode='w+', 
            shape=(dataset_len[args.dataset], args.dim)) 
    else:
        filename = os.path.join('{}/test-{}.npy'.format(
            args.save_dir, args.dim
        ))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        dstore_keys = np.memmap(filename, 
            dtype=np.float32, 
            mode='w+', 
            shape=(testset_len[args.dataset], args.dim)) 

    for batch_idx, batch in enumerate(loader):
        print(batch_idx)

        inputs, labels = batch["input"].to(device), batch["label"].to(device)

        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(
            output_function, has_aux=False, argnums=1
        )

        # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
        grads = torch.func.vmap(
            grads_loss,
            in_dims=(None, None, None, *([0] * len(batch))),
            randomness="different",
        )(model, func_weights, func_buffers, inputs, labels)

        project_grad = projector.project(grads, model_id=0)
        normalize_grad = project_grad / normalize_factor

        # save gradient
        index_start = batch_idx * args.batch_size
        index_end = index_start + args.batch_size
        while (np.abs(dstore_keys[index_start:index_end, 0:32]).sum()==0):
            dstore_keys[index_start:index_end] = normalize_grad.detach().cpu().numpy()

if __name__ == '__main__':
    args = parseArgs()
    main(args)
