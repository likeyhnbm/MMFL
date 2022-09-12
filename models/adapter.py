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
from copy import deepcopy

def build_adapter_model(basic_model, num_classes=2, reducation_factor=8):

    model = Adapter_ViT(basic_model=basic_model, reducation_factor=reducation_factor, num_classes=num_classes
        )

    return model

class Adapter_ViT(nn.Module):
    def __init__(self, basic_model,
        reducation_factor=8, num_classes=100):

        super(Adapter_ViT, self).__init__()

        self.basic_model = deepcopy(basic_model)
        self.embed_dim = embed_dim = self.basic_model.embed_dim
        
        self.adapter_downsample = nn.Linear(
                embed_dim,
                embed_dim // reducation_factor
            )
        self.adapter_upsample = nn.Linear(
                embed_dim // reducation_factor,
                embed_dim
            )
        self.adapter_act_fn = nn.functional.gelu

        nn.init.zeros_(self.adapter_downsample.weight)
        nn.init.zeros_(self.adapter_downsample.bias)

        nn.init.zeros_(self.adapter_upsample.weight)
        nn.init.zeros_(self.adapter_upsample.bias)


        # Change num_class
        self.basic_model.head = nn.Linear(self.embed_dim, num_classes)

        self._freeze()

        pass

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.adapter_downsample.parameters():
            param.requires_grad = True
        for param in self.adapter_upsample.parameters():
            param.requires_grad = True
        for param in self.basic_model.head.parameters():
            param.requires_grad = True
        

    def forward_features(self, x):
        
        x = self.basic_model.patch_embed(x)
        x = self.basic_model._pos_embed(x)

        for i in range(len(self.basic_model.blocks)):
            
            # forward normal blocks
            block = self.basic_model.blocks[i]
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            h = x
            x = block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
            # adapter
            adpt = self.adapter_downsample(x)
            adpt = self.adapter_act_fn(adpt)
            adpt = self.adapter_upsample(adpt)
            x = adpt + x

            x = x + h


        x = self.basic_model.norm(x)
        return x

    def forward(self,x):
        self.basic_model.eval()
        x = self.forward_features(x)
        x = self.basic_model.forward_head(x)
        return x
    
    def state_dict(self):

        state_dict = {}

        for k,p in self.named_parameters():
            if p.requires_grad:
                state_dict.update({k:p.data})

        
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        # self.head.load_state_dict(prompt_state_dict['head'], strict=strict)
        super().load_state_dict(state_dict, strict=False)