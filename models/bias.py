"""
   Source: https://github.com/sagizty/VPT/blob/main/PromptModels/structure.py
"""

from sqlite3 import adapt
import torch
import torch.nn as nn
import copy
import math
from functools import reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
import timm
from copy import deepcopy

def build_bias_model(type, num_classes=2):

    model = Bias_ViT(type, num_classes=num_classes)

    return model

class Bias_ViT(nn.Module):
    def __init__(self, type, num_classes=100):

        super(Bias_ViT, self).__init__()

        self.model = timm.create_model(type,num_classes=num_classes,pretrained=True)

        self._freeze()

        pass

    def _freeze(self):
        for k, p in self.model.named_parameters():
            if 'bias' not in k:
                p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True

    def forward(self,x):

        return self.model(x)
    
    def state_dict(self):

        state_dict = {}

        for k,p in self.named_parameters():
            if p.requires_grad:
                state_dict.update({k:p.data})

        
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        # self.head.load_state_dict(prompt_state_dict['head'], strict=strict)
        super().load_state_dict(state_dict, strict=False)