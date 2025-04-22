from PIL.Image import Image
from torchvision import transforms
import torch
import random

class AddGaussianNoise(object):
    """添加高斯噪声的转换类"""

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomColorAdjustment(transforms.ColorJitter):
    """随机调整图像的色调、饱和度、亮度、对比度"""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(RandomColorAdjustment, self).__init__(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        img = super().__call__(img)
        return img


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        RandomColorAdjustment(brightness=0.05, contrast=(0.8, 1.2), saturation=(0.5, 1.5), hue=0.08),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=3),  # 类似于zoom_range
        AddGaussianNoise(0.0, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

img = Image.new('RGB', (100, 100), color = 'red')
transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.0, 0.05)
])
img_t = transform(img)
print(img_t)