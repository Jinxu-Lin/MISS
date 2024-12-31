import argparse
import numpy as np
import random
import os
import pickle
import torch
import torch.nn.functional as F

from Tools.Data import cifar2, cifar10, imagenet
from Tools.Models.resnet import resnet18
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


def parseArgs():

    parser = argparse.ArgumentParser(
        description="Image Classification Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        dest="seed", help='seed')
    
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
    parser.add_argument("--batch-size", type=int, default=64, 
                        dest="batch_size", help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0,
                        dest="dataloader_num_workers", help="The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    
    # Model
    parser.add_argument("--model", type=str, default="resnet9",
                        dest="model", help="Model to use")
    parser.add_argument("--model-dir", type=str, default='./saved/models',
                        dest="model_dir", help='model directory')
    parser.add_argument("--model-name", type=str, default='model_10.pth',
                        dest="model_name", help='model name')               
    
    # Save
    parser.add_argument("--save-dir", type=str, default='./saved',
                        dest="save_dir", help='save directory')

    return parser.parse_args()


def main(args):

    # seed
    set_seed(args.seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

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

    batch_error_list = []

    for batch_idx, batch in enumerate(loader):
        print("batch_idx: ", batch_idx)
        data, labels = batch["input"].to(device), batch["label"].to(device)
        outputs = model(data)
        prob = F.softmax(outputs, dim=-1)
        conf, _ = torch.max(prob, dim=-1)
        
        error = 1 - conf
        batch_error_list.append(error.detach().cpu().numpy())

    save_name = f"{args.dataset_split}_error.pkl"
    with open(os.path.join(args.save_dir, save_name), "wb") as f:
        pickle.dump(batch_error_list, f)

if __name__ == "__main__":
    args = parseArgs()
    main(args)