"""
   Source: https://github.com/sagizty/VPT/blob/main/PromptModels/structure.py
"""

import torch
import torch.nn as nn
import copy

from timm.models.vision_transformer import VisionTransformer, PatchEmbed

def build_promptmodel(basic_model, num_classes=2, edge_size=384, patch_size=16,
                      Prompt_Token_num=10, VPT_type="Shallow"):
    # VPT_type = "Deep" / "Shallow"


    model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                    VPT_type=VPT_type,num_classes=num_classes,embed_dim=basic_model.embed_dim)

    model.load_state_dict(basic_model.state_dict(), False)    #TODO: Check it.
    model.New_CLS_head(num_classes)
    model.Freeze()

    # try:
    #     img = torch.randn(1, 3, edge_size, edge_size)
    #     preds = model(img)  # (1, class_number)
    #     print('test model output：', preds)
    # except:
    #     print("Problem exist in the model defining process！！")
    #     return -1
    # else:
    #     # print('model is ready now!')
    return model

    
class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=384, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', Prompt_Token_num=1, VPT_type="Shallow", basic_state_dict=None):

        # Recreate ViT
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                         representation_size, distilled, drop_rate, attn_drop_rate, drop_path_rate, embed_layer,
                         norm_layer, act_layer, weight_init)

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        else:  # "Shallow"
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))

    def New_CLS_head(self, new_classes=15):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Tokens.requires_grad = True
        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def obtain_prompt(self):

        prompt_state_dict = {k:v for k,v in self.state_dict().items() if k in ["head." + s for s in self.head.state_dict().keys()] or k=='Prompt_Tokens'}
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict, strict=False):
        # self.head.load_state_dict(prompt_state_dict['head'], strict=strict)
        self.load_state_dict(prompt_state_dict, strict=False)

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":

            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                # firstly concatenate
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            num_tokens = x.shape[1]
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            # Sequntially procees，lastly remove prompt tokens
            x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x

    def forward(self, x):

        x = self.forward_features(x)

        # use cls token for cls head
        x = self.pre_logits(x[:, 0, :])
        x = self.head(x)
        return x