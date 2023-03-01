import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from glob import glob
import numpy as np
class ImageNet1000_limit_images(data.Dataset):
    def __init__(self,root,split='train', 
                    transform=None, nb_images=83):
        super(ImageNet1000_limit_images, self).__init__()

        self.root = os.path.join(root, '%s' %(split))
        self.transform = transform
        self.split = split
#         if split=='train':
        self.nb_images = nb_images
        subdirs = glob(self.root+"/*/")

        # Gather the files (sorted)
        imgs = []
        self.targets = []
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            
            len_files = len(files)
            limit_nb_images = min(len_files,self.nb_images)
            
            files = files[:limit_nb_images]
            
            for f in files:
                imgs.append((f, i))
            self.targets.append(i)
#         print(subdir_path)
        self.imgs = imgs
        self.targets = np.array(self.targets)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size

        if self.transform is not None:
            img = self.transform(img)

        return img,target