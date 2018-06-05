import torch
import torch.nn as nn
from torchvision import models

VGG = models.vgg19(pretrained=True).features.eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class vgg16_features(nn.Module):
    def __init__(self):
        super(vgg16_features, self).__init__()
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)
        self.f1 = nn.Sequential(normalization)
        self.f2 = nn.Sequential()
        self.f3 = nn.Sequential()
        self.f4 = nn.Sequential()
        self.f5 = nn.Sequential()

        i = 1  # increment every time we see a conv
        j = 1
        is_pool = 0
        for layer in VGG.children():
            if isinstance(layer, nn.Conv2d):
                name = 'conv_{}_{}'.format(i, j)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}_{}'.format(i, j)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                is_pool = 1
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}_{}'.format(i, j)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            if i == 1:
                self.f1.add_module(name, layer)
            elif i == 2:
                self.f2.add_module(name, layer)
            elif i == 3:
                self.f3.add_module(name, layer)
            elif i == 4:
                self.f4.add_module(name, layer)
            else:
                self.f5.add_module(name, layer)

            j += 1
            if is_pool:
                i += 1
                is_pool = 0
                j = 0

    def forward(self, x):
        feature1 = self.f1(x)
        feature2 = self.f2(feature1)
        feature3 = self.f3(feature2)
        feature4 = self.f4(feature3)
        feature5 = self.f5(feature4)

        return [feature1, feature2, feature3, feature4, feature5]


class vgg16_deconvnet(nn.Module):
    def __init__(self):
        super(vgg16_deconvnet, self).__init__()
        self.f5 = nn.Sequential(

        )
        self.f4 = nn.Sequential(

        )