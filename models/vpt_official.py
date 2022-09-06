"""
   Source: https://github.com/sagizty/VPT/blob/main/PromptModels/structure.py
"""

import torch
import torch.nn as nn
import copy
import math
from functools import reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
from copy import deepcopy

def build_promptmodel(basic_model, num_classes=2, edge_size=384, patch_size=16,
                      prompt_num=10, vpt_type="Shallow", projection = 512,
                      prompt_drop_rate=0.1):
    # VPT_type = "Deep" / "Shallow"


    # model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
    #                 num_heads=num_heads,
    #                 VPT_type=VPT_type,num_classes=num_classes,embed_dim=basic_model.embed_dim, 
    #                 projection=projection,prompt_drop_rate=prompt_drop_rate)

    # model.load_state_dict(basic_model.state_dict(), False)    #TODO: Check it.
    # model.New_CLS_head(num_classes)
    # # model.Freeze()

    model = VPT_ViT(basic_model=basic_model, prompt_num=prompt_num,vpt_type=vpt_type,patch_size=patch_size,
        num_classes=num_classes, prompt_drop_rate=prompt_drop_rate, projection=projection
        )

    return model

    
# class VPT_ViT(VisionTransformer):
#     # def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=384, depth=12,
#     #              num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
#     #              drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
#     #              act_layer=None, weight_init='', Prompt_Token_num=1, VPT_type="Shallow", basic_state_dict=None, 
#     #              projection=512, prompt_drop_rate=0.1):

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
#             embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
#             class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
#             weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, 
#             Prompt_Token_num=1, VPT_type="Shallow", basic_state_dict=None, 
#                  projection=512, prompt_drop_rate=0.1):

#         # Recreate ViT
#         super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
#             embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
#             class_token=class_token, no_embed_class=no_embed_class, fc_norm=fc_norm, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
#             weight_init=weight_init, embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)

#         # load basic state_dict
#         if basic_state_dict is not None:
#             self.load_state_dict(basic_state_dict, False)

#         self.VPT_type = VPT_type

#         if projection > -1:
#             # only for prepend / add
#             prompt_dim = projection
#             self.prompt_proj = nn.Linear(
#                 prompt_dim, embed_dim)
#             nn.init.kaiming_normal_(
#                 self.prompt_proj.weight, a=0, mode='fan_out')
#         else:
#             prompt_dim = embed_dim
#             self.prompt_proj = nn.Identity()

#         val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + prompt_dim))  # noqa


#         if VPT_type == "Deep":
#             self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, prompt_dim))
#             # xavier_uniform initialization
#             nn.init.uniform_(self.Prompt_Tokens.data, -val, val)

#         else:  # "Shallow"
#             self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, prompt_dim))
#             nn.init.uniform_(self.Prompt_Tokens.data, -val, val)

#         self.prompt_dropout = nn.Dropout(prompt_drop_rate)


#     def New_CLS_head(self, new_classes=15):
#         self.head = nn.Linear(self.embed_dim, new_classes)

#     def Freeze(self):
#         for param in self.parameters():
#             param.requires_grad = False

#         self.Prompt_Tokens.requires_grad = True
        
#         try:
#             for param in self.head.parameters():
#                 param.requires_grad = True
#             for param in self.prompt_proj.parameters():
#                 param.requires_grad = True
#         except:
#             pass

#     def obtain_prompt(self):

#         prompt_state_dict = {k:v for k,v in self.state_dict().items() if k in ["head." + s for s in self.head.state_dict().keys()] or k=='Prompt_Tokens' or k in ["prompt_proj." + s for s in self.prompt_proj.state_dict().keys()]}
#         return prompt_state_dict

#     def load_prompt(self, prompt_state_dict, strict=False):
#         # self.head.load_state_dict(prompt_state_dict['head'], strict=strict)
#         self.load_state_dict(prompt_state_dict, strict=False)

#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         # print(x.shape,self.pos_embed.shape)
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)

#         # concatenate CLS token
#         x = torch.cat((cls_token, x), dim=1)
#         x = self.pos_drop(x + self.pos_embed)

#         if self.VPT_type == "Deep":

#             Prompt_Token_num = self.Prompt_Tokens.shape[1]

#             for i in range(len(self.blocks)):
#                 # concatenate Prompt_Tokens
#                 Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
#                 Prompt_Tokens = self.prompt_dropout(self.prompt_proj(Prompt_Tokens))
#                 # firstly concatenate
#                 x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
#                 num_tokens = x.shape[1]
#                 # lastly remove, a genius trick
#                 x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

#         else:  # self.VPT_type == "Shallow"
            
#             Prompt_Token_num = self.Prompt_Tokens.shape[1]
#             # concatenate Prompt_Tokens
#             Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
#             Prompt_Tokens = self.prompt_dropout(self.prompt_proj(Prompt_Tokens))
#             x = torch.cat((x, Prompt_Tokens), dim=1)
#             num_tokens = x.shape[1]
#             # Sequntially procees，lastly remove prompt tokens
#             x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

#         x = self.norm(x)
#         return x

#     # def forward(self, x):

#     #     x = self.forward_features(x)

#     #     # use cls token for cls head
#     #     x = self.pre_logits(x[:, 0, :])
#     #     x = self.head(x)
#     #     return x


class VPT_ViT(nn.Module):
    def __init__(self, basic_model,
        prompt_num=1, vpt_type="Shallow", patch_size=16, depth=12, num_classes=1000,
        projection=-1, prompt_drop_rate=0.1):

        super(VPT_ViT, self).__init__()

        self.basic_model = deepcopy(basic_model)
        self.embed_dim = embed_dim = self.basic_model.embed_dim
        self.prompt_num = prompt_num

        # Change num_class
        self.basic_model.head = nn.Linear(self.embed_dim, num_classes)

        self.vpt_type = vpt_type

        if projection > -1:
            # only for prepend / add
            prompt_dim = projection
            self.prompt_proj = nn.Linear(
                prompt_dim, embed_dim)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = embed_dim
            self.prompt_proj = nn.Identity()

        val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + prompt_dim))  # noqa


        if vpt_type == "Deep":
            self.prompt_embeddings = nn.Parameter(torch.zeros(depth, prompt_num, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        else:  # "Shallow"
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, prompt_num, prompt_dim))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        self.prompt_dropout = nn.Dropout(prompt_drop_rate)

        self._freeze()

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.prompt_embeddings.requires_grad = True
        for param in self.basic_model.head.parameters():
            param.requires_grad = True
        for param in self.prompt_proj.parameters():
            param.requires_grad = True
        

    def forward_features(self, x):
        
        x = self.basic_model.patch_embed(x)
        x = self.basic_model._pos_embed(x)

        if self.vpt_type == "Deep":

            for i in range(len(self.basic_model.blocks)):
                # concatenate Prompt_Tokens
                prompt_embeddings = self.prompt_embeddings[i].unsqueeze(0)
                prompt_embeddings = self.prompt_dropout(self.prompt_proj(prompt_embeddings))
                # firstly concatenate
                x = torch.cat((x, prompt_embeddings.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x = self.basic_model.blocks[i](x)[:, :num_tokens - self.prompt_num]

        else:  # self.VPT_type == "Shallow"
            
            # concatenate Prompt_Tokens
            prompt_embeddings = self.prompt_embeddings.expand(x.shape[0], -1, -1)
            prompt_embeddings = self.prompt_dropout(self.prompt_proj(prompt_embeddings))
            x = torch.cat((x, prompt_embeddings), dim=1)
            num_tokens = x.shape[1]
            # Sequntially procees，lastly remove prompt tokens
            x = self.basic_model.blocks(x)[:, :num_tokens - self.prompt_num]

        x = self.basic_model.norm(x)
        return x

    def forward(self,x):
        self.basic_model.eval()
        x = self.forward_features(x)
        x = self.basic_model.forward_head(x)
        return x
    
    def obtain_prompt(self):

        prompt_state_dict = {k:v for k,v in self.state_dict().items() if k in ["head." + s for s in self.head.state_dict().keys()] or k=='Prompt_Tokens' or k in ["prompt_proj." + s for s in self.prompt_proj.state_dict().keys()]}
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict, strict=False):
        # self.head.load_state_dict(prompt_state_dict['head'], strict=strict)
        self.load_state_dict(prompt_state_dict, strict=False)
