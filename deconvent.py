import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from collections import OrderedDict
VGG = models.vgg16(pretrained=True).features.eval()

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

class Denormalization(nn.Module):
    def __init__(self, mean, std):
        super(Denormalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # denormalize img
        return img * self.std + self.mean

class vgg16_features(nn.Module):
    def __init__(self):
        super(vgg16_features, self).__init__()
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)
        self.f1 = nn.Sequential()
        self.f2 = nn.Sequential()
        self.f3 = nn.Sequential()
        self.f4 = nn.Sequential()
        self.f5 = nn.Sequential()

        self.f1.add_module("norm_1_0",normalization)

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
                layer = nn.MaxPool2d(2,stride=2,return_indices=True)
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

        self.pool_indices = dict()

    def forward_features(self, x,f,s):
        output = x
        for layer in f:
            if isinstance(layer, torch.nn.MaxPool2d):
                output, indices = layer(output)
                self.pool_indices[s] = indices
            else:
                output = layer(output)
        return output

    def forward(self, x):
        feature1 = self.forward_features(x,self.f1,1)
        feature2 = self.forward_features(feature1,self.f2,2)
        feature3 = self.forward_features(feature2,self.f3,3)
        feature4 = self.forward_features(feature3,self.f4,4)
        feature5 = self.forward_features(feature4,self.f5,5)

        return [feature1, feature2, feature3, feature4, feature5], self.pool_indices


class vgg16_deconvnet(nn.Module):
    def __init__(self):
        super(vgg16_deconvnet, self).__init__()
        denormalization = Denormalization(cnn_normalization_mean, cnn_normalization_std)
        self.f5 = nn.Sequential()
        self.f4 = nn.Sequential()
        self.f3 = nn.Sequential()
        self.f2 = nn.Sequential()
        self.f1 = nn.Sequential()

        self.f5.add_module("unpool_5", torch.nn.MaxUnpool2d(2, stride=2))
        self.f5.add_module("relu_5_5", torch.nn.ReLU(inplace=False))
        self.f5.add_module("conv_5_4", torch.nn.ConvTranspose2d(512, 512, 3, padding=1))
        self.f5.add_module("relu_5_3", torch.nn.ReLU(inplace=False))
        self.f5.add_module("conv_5_2", torch.nn.ConvTranspose2d(512, 512, 3, padding=1))
        self.f5.add_module("relu_5_1", torch.nn.ReLU(inplace=False))
        self.f5.add_module("conv_5_0",  torch.nn.ConvTranspose2d(512, 512, 3, padding=1))

        self.f4.add_module("unpool_4", torch.nn.MaxUnpool2d(2, stride=2))
        self.f4.add_module("relu_4_5", torch.nn.ReLU(inplace=False))
        self.f4.add_module("conv_4_4", torch.nn.ConvTranspose2d(512, 512, 3, padding=1))
        self.f4.add_module("relu_4_3", torch.nn.ReLU(inplace=False))
        self.f4.add_module("conv_4_2", torch.nn.ConvTranspose2d(512, 512, 3, padding=1))
        self.f4.add_module("relu_4_1", torch.nn.ReLU(inplace=False))
        self.f4.add_module("conv_4_0", torch.nn.ConvTranspose2d(512, 256, 3, padding=1))

        self.f3.add_module("unpool_3", torch.nn.MaxUnpool2d(2, stride=2))
        self.f3.add_module("relu_3_5", torch.nn.ReLU(inplace=False))
        self.f3.add_module("conv_3_4", torch.nn.ConvTranspose2d(256, 256, 3, padding=1))
        self.f3.add_module("relu_3_3", torch.nn.ReLU(inplace=False))
        self.f3.add_module("conv_3_2", torch.nn.ConvTranspose2d(256, 256, 3, padding=1))
        self.f3.add_module("relu_3_1", torch.nn.ReLU(inplace=False))
        self.f3.add_module("conv_3_0", torch.nn.ConvTranspose2d(256, 128, 3, padding=1))

        self.f2.add_module("unpool_2", torch.nn.MaxUnpool2d(2, stride=2))
        self.f2.add_module("relu_2_3", torch.nn.ReLU(inplace=False))
        self.f2.add_module("conv_2_2", torch.nn.ConvTranspose2d(128, 128, 3, padding=1))
        self.f2.add_module("relu_2_1", torch.nn.ReLU(inplace=False))
        self.f2.add_module("conv_2_0", torch.nn.ConvTranspose2d(128, 64, 3, padding=1))

        self.f1.add_module("unpool_1", torch.nn.MaxUnpool2d(2, stride=2))
        self.f1.add_module("relu_1_4", torch.nn.ReLU(inplace=False))
        self.f1.add_module("conv_1_3", torch.nn.ConvTranspose2d(64, 64, 3, padding=1))
        self.f1.add_module("relu_1_2", torch.nn.ReLU(inplace=False))
        self.f1.add_module("conv_1_1", torch.nn.ConvTranspose2d(64, 3, 3, padding=1))
        self.f1.add_module("norm_1_0", denormalization)

        self._initialize_weights()

    def _initialize_weights(self):
        self.f1[4].weight.data = VGG[0].weight.data
        self.f1[2].weight.data = VGG[2].weight.data
        self.f2[4].weight.data = VGG[5].weight.data
        self.f2[2].weight.data = VGG[7].weight.data
        self.f3[6].weight.data = VGG[10].weight.data
        self.f3[4].weight.data = VGG[12].weight.data
        self.f3[2].weight.data = VGG[14].weight.data
        self.f4[6].weight.data = VGG[17].weight.data
        self.f4[4].weight.data = VGG[19].weight.data
        self.f4[2].weight.data = VGG[21].weight.data
        self.f5[6].weight.data = VGG[24].weight.data
        self.f5[4].weight.data = VGG[26].weight.data
        self.f5[2].weight.data = VGG[28].weight.data

        self.f1[4].bias.data = torch.zeros(self.f1[4].bias.data.shape)
        self.f1[2].bias.data = VGG[0].bias.data
        self.f2[4].bias.data = VGG[2].bias.data
        self.f2[2].bias.data = VGG[5].bias.data
        self.f3[6].bias.data = VGG[7].bias.data
        self.f3[4].bias.data = VGG[10].bias.data
        self.f3[2].bias.data = VGG[12].bias.data
        self.f4[6].bias.data = VGG[14].bias.data
        self.f4[4].bias.data = VGG[17].bias.data
        self.f4[2].bias.data = VGG[19].bias.data
        self.f5[6].bias.data = VGG[21].bias.data
        self.f5[4].bias.data = VGG[24].bias.data
        self.f5[2].bias.data = VGG[26].bias.data



    def forward_features(self, f, x, s):
        output = x
        for layer in f:
            if isinstance(layer, torch.nn.MaxUnpool2d):
                output = layer(output, s)
            else:
                output = layer(output)
        return output

    def forward(self, x, pool_indices):
        feature5 = self.forward_features(self.f5, x, pool_indices[5])
        feature4 = self.forward_features(self.f4, feature5, pool_indices[4])
        feature3 = self.forward_features(self.f3, feature4, pool_indices[3])
        feature2 = self.forward_features(self.f2, feature3, pool_indices[2])
        feature1 = self.forward_features(self.f1, feature2, pool_indices[1])
        return [feature5, feature4, feature3, feature2, feature1]


if __name__ == '__main__':
    VGG_f = vgg16_features()
    VGG_de = vgg16_deconvnet()




