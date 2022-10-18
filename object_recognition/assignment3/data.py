import zipfile
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional

# Padding calculation (unused)
def get_padding(image):
    imsize = image.size
    max_h = max_w = max(imsize[0], imsize[1])
 
    h_padding = (max_w - imsize[0]) / 2
    v_padding = (max_h - imsize[1]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


def pad_image(self, image):
        padded_im =  torchvision.transforms.functional.pad(image, get_padding(image)) # torchvision.transforms.functional.pad
        return padded_im


class NewPad(object):
    # def __init__(self, fill=0, padding_mode='constant'):
    def __init__(self):
        print("pad")

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return torchvision.transforms.functional.pad(img, get_padding(img))
    
    def __repr__(self):
        return self.__class__.__name__ 

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



# Data transforms pipeline (unused is commented)

square_size = 224
# square_size = 299
resize = int(0.8 * square_size)

data_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(resize, scale=(0.8,0.8)),
    # transforms.RandomCrop((resize,resize)),
    # transforms.RandomCrop((square_size, square_size)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
	transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.4, hue=0), 
	# transforms.GaussianBlur(5),
	# AddGaussianNoise(mean = 0, std = 0.5),
    transforms.RandomRotation((90,90)),
	# torchvision.transforms.ColorJitter(),
    transforms.RandomPerspective(distortion_scale = 0.3,p = 0.3),
    # NewPad(),
    transforms.Resize((square_size, square_size)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])


data_transforms_test = transforms.Compose([
    transforms.Resize((square_size, square_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
