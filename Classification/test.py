import os
from datasets import load_from_disk
from torchvision import transforms
import torch

dataset = load_from_disk(
    os.path.join("/home/jinxulin/MISS/CIFAR10", "train")
)

augmentations = transforms.Compose([
    transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
])

def compute_mean_std(dataset, augmentations):
    all_pixels = []
    for example in dataset:
        # 应用数据增强，确保图像已转换为张量
        image = augmentations(example["img"].convert("RGB"))
        all_pixels.append(image)

    # 将所有图像堆叠为一个大的张量
    all_pixels = torch.stack(all_pixels)

    # 计算每个通道的均值和标准差
    mean = all_pixels.mean(dim=[0, 2, 3])  # 按通道计算
    std = all_pixels.std(dim=[0, 2, 3])    # 按通道计算
    return mean, std

mean, std = compute_mean_std(dataset, augmentations)
print(mean, std)