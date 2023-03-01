import torchvision.transforms as T
import numpy as np
import PIL.Image as Image
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    T.GaussianBlur = GaussianBlur
    
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

def get_mask_image_pair(img_full,h_high,h_low,w_high,w_low,overlap=0.5):
    img_full = np.array(img_full)
    hi,wi,ci = img_full.shape
    h = np.random.uniform(low=h_low, high=h_high, size=(1,))[0]
    w = np.random.uniform(low=w_low, high=w_high, size=(1,))[0]
    
    w = h.copy()

    x = np.random.uniform(low=0, high=1-h, size=(1,))[0]

    y = np.random.uniform(low=0, high=1-w, size=(1,))[0]
    
    
    
    overlapX = overlap*x
    overlapY = overlap*y
    ho_ = int(overlapX*hi/2)
    wo_ = int(overlapY*wi/2)

    # h,w,x,y = 0.4,0.2,0.4,0.2
    
#     h+=0.1
#     w=+=0.1
#     x+=0.1
#     y+=0.1
    h_,w_,x_,y_ = int(h*hi),int(w*wi),int(x*hi),int(y*wi)
    
    ho_ = min(ho_,x_)
    wo_ = min(wo_,y_)

    if np.random.random()>0:
        x1 = img_full[x_:x_+h_,y_:y_+w_].copy()
    else:
        x1 = img_full.copy()
    x2 = img_full
    x2[x_+ho_:x_+h_-ho_,y_+wo_:y_+w_-wo_] = 0 #np.random.randn(*x1.shape)*255
    x1 = Image.fromarray(x1)
    x2 = Image.fromarray(x2)
    return (x1,x2)

def get_non_overlapping_image_pair(img_full,x_high,x_low,y_high,y_low):
    img_full = np.array(img_full)
    hi,wi,ci = img_full.shape
    
    x = np.random.uniform(low=x_low, high=x_high, size=(1,))[0]
    y = np.random.uniform(low=y_low, high=y_high, size=(1,))[0]

    x_,y_ = int(x*hi),int(y*wi)

    x1 = img_full[:x_,:y_].copy()
    select_section = np.random.randint(3)
    if select_section == 0:
        x2 = img_full[x_:,:y_]
    elif select_section == 1:
        x2 = img_full[x_:,y_:]
    elif select_section == 2:
        x2 = img_full[:x_,y_:]
#     x2 = img_full
#     x2[x_:x_+h_,y_:y_+w_] = np.random.randn(*x1.shape)*255
    x1 = Image.fromarray(x1)
    x2 = Image.fromarray(x2)
    return (x1,x2)

def get_with_overlapping_image_pair(img_full,x_high,x_low,y_high,y_low,overlap=0.5):
    img_full = np.array(img_full)
    hi,wi,ci = img_full.shape
    
    x = np.random.uniform(low=x_low, high=x_high, size=(1,))[0]
    y = np.random.uniform(low=y_low, high=y_high, size=(1,))[0]
#     overlap = 0.1
    overlapX = overlap*x
    overlapY = overlap*y
    ho_ = int(overlapX*hi)
    wo_ = int(overlapY*wi)

#     x = np.random.uniform(low=h, high=h+0.2, size=(1,))[0]

#     y = np.random.uniform(low=w, high=w+0.2, size=(1,))[0]

    # h,w,x,y = 0.4,0.2,0.4,0.2
#     print(x,y)
#     h_,w_,x_,y_ = int(h*hi),int(w*wi),int(x*hi),int(y*wi)
    x_,y_ = int(x*hi),int(y*wi)
    
    ho_ = min(ho_,x_)
    wo_ = min(wo_,y_)

    select_list = [0,1,2,3]
    np.random.shuffle(select_list)
    
#     select_section = np.random.randint(3)
    x1x2 = []
    for i in range(0,2):
#         print(select_list[i])
        if select_list[i] == 0:
            x1x2.append(img_full[x_-ho_:,:y_+wo_])
        elif select_list[i] == 1:
            x1x2.append(img_full[x_-ho_:,y_-wo_:])
        elif select_list[i] == 2:
            x1x2.append(img_full[:x_+ho_,y_-wo_:])
        elif select_list[i] == 3:
            x1x2.append(img_full[:x_+ho_,:y_+wo_].copy())
    x1 = x1x2[0]
    x2 = x1x2[1]
        
#     x2 = img_full
#     x2[x_:x_+h_,y_:y_+w_] = np.random.randn(*x1.shape)*255
    x1 = Image.fromarray(x1)
    x2 = Image.fromarray(x2)
    return (x1,x2)

class SimSiamTransform():
    def __init__(self, image_size, overlap_size, mean_std=imagenet_mean_std):
        self.overlap_size = overlap_size
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform_orig = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.transform_modified = T.Compose([
            T.RandomResizedCrop(image_size, scale=(1.0, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
#         print(x.size)
#         x1,x2 = get_mask_image_pair(x,h_high=0.4,h_low=0.2,w_high=0.4,w_low=0.2)
        if np.random.random()>0:
            x1,x2 = get_with_overlapping_image_pair(x,x_high=0.5,x_low=0.2,y_high=0.5,y_low=0.2,overlap=self.overlap_size)
#             x1,x2 = get_mask_image_pair(x,h_high=0.4,h_low=0.2,w_high=0.4,w_low=0.2,overlap=0.5)
            x1 = self.transform_modified(x1)
            x2 = self.transform_modified(x2)
#         else:
#             x1 = self.transform_orig(x)
#             x2 = self.transform_orig(x) 
#         x1 = self.transform_orig(x)
#         x2 = self.transform_orig(x)
#         print(x1.size,x2.size)
        
#         print(x1.shape,x2.shape)
        return x1, x2 


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    if isinstance(pic, torch.Tensor):
        if pic.is_floating_point() and mode != 'F':
            pic = pic.mul(255).byte()
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)










