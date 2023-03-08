import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from glob import glob
import numpy as np
class ImageNet100_aug_refine(data.Dataset):
    def __init__(self, subset_file, root, split='train', 
                    transform=None, r_model=None, c_model=None, r_c_loss_avg = None):
        super(ImageNet100_aug_refine, self).__init__()

        self.root = os.path.join(root, '%s' %(split))
        self.transform = transform
        self.split = split
        self.r_model = r_model
        self.c_model = c_model
        self.r_c_loss_avg = r_c_loss_avg
        self.gpu = None

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs = []
        for line in result:
#             print(line)
            subdir = line
#             print(subdir)
            
            subdirs.append(subdir)

        # Gather the files (sorted)
        imgs = []
        self.targets = []
        self.classes = []
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.jpg')))
            
            for f in files:
                imgs.append((f, i))
                self.targets.append(i)
            self.classes.append(i)
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
        
        if self.r_model and self.c_model and self.r_c_loss_avg is not None:
            with torch.no_grad():
                print("here")
                c_img = self.c_model(img[0].unsqueeze(0).cuda(self.gpu, non_blocking=True))
                r_img = self.r_model(img[0].unsqueeze(0).cuda(self.gpu, non_blocking=True))
                loss = torch.nn.functional.mse_loss(c_img.detach(), r_img.detach())
                if self.r_c_loss_avg > loss:
                    img, target = self.__getitem__(np.random.randint(0, len(self.imgs)))
        return img,target
