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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


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
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad==True}
    buffers = {k: v.detach() for k, v in model.named_buffers() if v.requires_grad==True}

    # initialize projector
    projector = CudaProjector(
        grad_dim=count_parameters(model), 
        proj_dim=args.dim,
        seed=args.seed, 
        proj_type=ProjectionType.normal,
        device=device,
        max_batch_size=args.batch_size
    )

    # Initialize save np array
    if args.dataset_split == 'train':
        filename = os.path.join('{}/{}/train-grad-{}-{}-{}.npy'.format(
            args.save_dir, args.dataset, args.model, args.model_name, args.dim
        ))
    else:
        filename = os.path.join('{}/{}/test-grad-{}-{}-{}.npy'.format(
            args.save_dir, args.dataset, args.model, args.model_name, args.dim
        ))
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # TODO: test store shape not defined
    dstore_keys = np.memmap(filename, 
                            dtype=np.float32, 
                            mode='w+', 
                            shape=(dataset_len[args.dataset], args.dim)) 
    
    
    # model evaluation for gradient computation
    # define model output function
    def compute_f(params, buffers, inputs, labels):
        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)
    
        predictions = functional_call(model, (params, buffers), args=inputs)
        ####
        f = torch.nn.CrossEntropyLoss()(predictions, labels)
        ####
        return f 


    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(
        ft_compute_grad, 
        in_dims=(None, None, 0, 0)
    )

    for batch_idx, batch in enumerate(loader):
        print("batch_idx: ", batch_idx)
        inputs, labels = batch["input"].to(device), batch["label"].to(device)

        # compute gradient
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, inputs, labels)
        ft_per_sample_grads = vectorize_and_ignore_buffers(list(ft_per_sample_grads.values()))

        # project gradient
        ft_per_sample_grads = projector.project(ft_per_sample_grads, model_id=0)

        # save gradient
        index_start = batch_idx * args.batch_size
        index_end = index_start + args.batch_size
        while (np.abs(dstore_keys[index_start:index_end, 0:32]).sum()==0):
            dstore_keys[index_start:index_end] = ft_per_sample_grads.detach().cpu().numpy()


if __name__ == "__main__":
    args = parseArgs()
    main(args)
