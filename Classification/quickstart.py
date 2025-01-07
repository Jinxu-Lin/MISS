
import os
from pathlib import Path
import wget
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
import warnings
from trak import TRAKer
from torch.nn import Module
from torch import Tensor
from typing import Iterable
from torch.func import functional_call, vmap, grad 

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


# Resnet9
class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)


def construct_rn9(num_classes=10):
    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
        return torch.nn.Sequential(
                torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=groups, bias=False),
                torch.nn.BatchNorm2d(channels_out),
                torch.nn.ReLU(inplace=True)
        )
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2)
    )
    return model


def load_grad(traker, model_id, device):
    traker.saver.load_current_store(model_id)

    g = torch.as_tensor(traker.saver.current_store["features"], device=device)
    return g


def get_dataloader(batch_size=256, num_workers=8, split='train', shuffle=False, augment=True):
    if augment:
        transforms = torchvision.transforms.Compose(
                        [torchvision.transforms.RandomHorizontalFlip(),
                         torchvision.transforms.RandomAffine(0),
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])
    else:
        transforms = torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])
        
    is_train = (split == 'train')
    dataset = torchvision.datasets.CIFAR10(root='/tmp/cifar/',
                                           download=True,
                                           train=is_train,
                                           transform=transforms)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size,
                                         num_workers=num_workers)
    
    return loader

def main():

    # device 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # load model
    base_path = Path('./Classification/saved/models/cifar10/origin')
    model_files = sorted(list(base_path.rglob('seed-0/model_23.pth')))
    ckpts = [torch.load(model_file, map_location='cpu') for model_file in model_files]
    model = construct_rn9().to(memory_format=torch.channels_last).cuda()
    if ckpts:
        model.load_state_dict(ckpts[-1])
        model = model.eval()
    else:
        print("No model files found!")

    func_weights = dict(model.named_parameters())
    func_buffers = dict(model.named_buffers())

    # load data
    batch_size = 128
    loader_train = get_dataloader(batch_size=batch_size, split='train')
    loader_targets = get_dataloader(batch_size=batch_size, split='val', augment=False)

    # set up trak
    traker = TRAKer(model=model,
                task='image_classification',
                proj_dim=4096,
                train_set_size=len(loader_train.dataset),
                save_dir='./Classification/trak_results')
    
    dstore_keys = np.memmap('./Classification/trak_results/train-4096.npy', 
        dtype=np.float16, 
        mode='w+', 
        shape=(50000, 4096))
    
    # compute trak features for train data
    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.load_checkpoint(ckpt, model_id=model_id)
        for batch in tqdm(loader_train):
            batch = [x.cuda() for x in batch]
            # compute_per_sample_grad & compute_loss_grad for a batch
            g = traker.featurize(batch=batch, num_samples=batch[0].shape[0])

            grads_loss = torch.func.grad(
                output_function, has_aux=False, argnums=1
            )

            # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
            grads = torch.func.vmap(
                grads_loss,
                in_dims=(None, None, None, *([0] * len(batch))),
                randomness="different",
            )(model, func_weights, func_buffers, batch[0], batch[1])

            
    traker.finalize_features()

    # compute trak features for test data
    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.start_scoring_checkpoint(exp_name='cifar10',
                                        checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=len(loader_targets.dataset))
        for batch in loader_targets:
            batch = [x.cuda() for x in batch]
            traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores(exp_name='cifar10')
    print(scores.shape)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()