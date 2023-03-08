import torch
import torchvision
from .random_dataset import RandomDataset
from .imagenet100 import ImageNet100
from .imagenet100_aug_refine import ImageNet100_aug_refine
import os
this_dir, this_filename = os.path.split(__file__)

def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None,ann_dir=None):
    print("data_dir", data_dir)
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    elif dataset == 'stl10_train':
        dataset = torchvision.datasets.STL10(data_dir, split='train' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    elif dataset == 'imagenet100':
#         print(os.path.join(this_dir, 'imagenet100.txt'))
        dataset = ImageNet100(subset_file=os.path.join(this_dir, 'imagenet100.txt'), root=data_dir, split='train' if train == True else 'val', transform=transform)
    elif dataset == 'imagenet100_aug_refine':
#         print(os.path.join(this_dir, 'imagenet100.txt'))
        dataset = ImageNet100_aug_refine(subset_file=os.path.join(this_dir, 'imagenet100.txt'), root=data_dir, split='train' if train == True else 'val', transform=transform)
    elif dataset == 'random':
        dataset = RandomDataset()
    elif dataset == 'mscoco':
        dataset = torchvision.datasets.CocoCaptions(root=data_dir, annFile=ann_dir, transform=transform)
    else:
        raise NotImplementedError
    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch

    return dataset