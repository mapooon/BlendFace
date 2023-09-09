import torch
import torch.nn.functional as F
from torch import nn

from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from model.generator import ADDGenerator
from model.encoder import MultilevelAttributesEncoder
from model.iresnet import iresnet100

import random
import numpy as np
import copy


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean 
        self.std = std

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x


class BlendSwap(nn.Module):
    def __init__(self):
        super(BlendSwap, self).__init__()
        self.G = ADDGenerator(512,3)
        self.E = MultilevelAttributesEncoder()

        Z_e = iresnet100(pretrained=False, fp16=False)
        Z_e=nn.Sequential(Normalize(0.5,0.5),Z_e)
        self.Z_e=Z_e

        self.mask_head=nn.Conv2d(64,1,1)

        
        self.G_ema=copy.deepcopy(self.G).eval()
        self.E_ema=copy.deepcopy(self.E).eval()
        self.mask_head_ema=copy.deepcopy(self.mask_head).eval()
            

    def forward(self,target_img,source_img):
        E=self.E_ema
        G=self.G_ema
        mask_head=self.mask_head_ema
        self.Z_e.eval()
        G.eval()
        E.eval()
        mask_head.eval()

        with torch.no_grad():
            z_id = self.Z_e(source_img)
            z_id = F.normalize(z_id)
            feature_map_t = E(target_img)
            output_g = G(z_id, feature_map_t)

            mask=mask_head(feature_map_t[-1]).sigmoid()
            output=output_g*mask+target_img*(1-mask)
        return output
