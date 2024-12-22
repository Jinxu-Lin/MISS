import argparse
import numpy as np
import random
import os

import torch
from torch import optim
from torch.func import functional_call, vmap, grad 

from trak.projectors import ProjectionType, AbstractProjector, CudaProjector

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


def compute_f(model, params, buffers, inputs, labels):
    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)
   
    predictions = functional_call(model, (params, buffers), args=inputs)
    ####
    f = torch.nn.CrossEntropyLoss()(predictions.float(), labels.float())
    ####
    return f  


def vectorize_and_ignore_buffers(g, params_dict=None):
    """
    gradients are given as a tuple :code:`(grad_w0, grad_w1, ... grad_wp)` where
    :code:`p` is the number of weight matrices. each :code:`grad_wi` has shape
    :code:`[batch_size, ...]` this function flattens :code:`g` to have shape
    :code:`[batch_size, num_params]`.
    """
    batch_size = len(g[0])
    out = []
    if params_dict is not None:
        for b in range(batch_size):
            out.append(torch.cat([x[b].flatten() for i, x in enumerate(g) if is_not_buffer(i, params_dict)]))
    else:
        for b in range(batch_size):
            out.append(torch.cat([x[b].flatten() for x in g]))
    return torch.stack(out)


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
    if 'idx-train.pkl' in args.index_path:
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
    model.load_state_dict(f'./models/{args.model_name}')

    # initialize projector
    projector = CudaProjector(
        grad_dim=count_parameters(model), 
        proj_dim=args.dim,
        seed=42, 
        proj_type=ProjectionType.normal,
        device='cuda:0'
    )

    # get params and buffers
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad==True}
    buffers = {k: v.detach() for k, v in model.named_buffers() if v.requires_grad==True}

    # Initialize save np array
    if 'idx-train.pkl' in args.index_path:
        filename = os.path.join('./saved/grad/train-grad-{}-{}-{}.npy'.format(
            args.model, args.model_name, args.dim
        ))
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    dstore_keys = np.memmap(filename, 
                            dtype=np.float32, 
                            mode='w+', 
                            shape=(dataset_len[args.dataset], args.dim)) 
    
    
    # model evaluation for gradient computation
    model.to(device)
    model.train()

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, 
                              in_dims=(None, None, 0, 0, 0, 
                                       ),
                             )

    for batch_idx, batch in enumerate(loader):
        print("batch_idx: ", batch_idx)
        inputs, labels = batch["input"].to(device), batch["label"].to(device)

        ft_per_sample_grads = ft_compute_sample_grad(model, params, buffers, inputs, labels)
        ft_per_sample_grads = vectorize_and_ignore_buffers(list(ft_per_sample_grads.values()))



if __name__ == "__main__":
    args = parseArgs()
    main(args)
