from .simsiam_aug import SimSiamTransform
from .simsiam_aug_overlap import SimSiamTransform as SimSiamTransform_overlap
from .simsiam_aug_adaptive_overlap import SimSiamTransform as SimSiamTransform_adaptive_overlap
# from .simsiam_aug_patch_style_augment import SimSiamTransform as SimSiamTransform_patch_style
from .simsiam_aug_patch import SimSiamTransform as SimSiamTransform_patch
from .simsiam_aug_cutout import SimSiamTransform as SimSiamTransform_cutout
from .simsiam_aug_cutout_blur import SimSiamTransform as SimSiamTransform_cutout_blur
from .simsiam_aug_tunnel import SimSiamTransform as SimSiamTransform_tunnel
from .simsiam_aug_spatial_hinge import SimSiamTransform as SimSiamTransform_spatial_hinge

from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
def get_aug(name, image_size=None, train=True, train_classifier=True,overlap_size=1.0,patch_size = 1.0,cutout_size=0.5,tunnel_size=0.5):

    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        elif name == 'simsiam_overlap':
            augmentation = SimSiamTransform_overlap(image_size,overlap_size)
        elif name == 'simsiam_adaptive_overlap':
            augmentation = SimSiamTransform_adaptive_overlap(image_size,overlap_size)
        elif name == 'simsiam_patch':
            augmentation = SimSiamTransform_patch(image_size,patch_size)
        # elif name == 'simsiam_patch_style':
        #     augmentation = SimSiamTransform_patch_style(image_size)
        elif name == 'simsiam_cutout':
            augmentation = SimSiamTransform_cutout(image_size,cutout_size)
        elif name == 'simsiam_cutout_blur':
            augmentation = SimSiamTransform_cutout_blur(image_size,cutout_size)
        elif name == 'simsiam_tunnel':
            augmentation = SimSiamTransform_tunnel(image_size,tunnel_size)
        elif name == 'simsiam_spatial_hinge':
            augmentation = SimSiamTransform_spatial_hinge(image_size)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        else:
            raise NotImplementedError
    elif train==False:
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








