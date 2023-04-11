""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models import create_model
# from pytorch_lightning.utilities.distributed import rank_zero_info
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))
        
        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_vlffn=False,
        layer_scale_init_values=0.1,
        max_text_len=40,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_text = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_imag = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_vl = None
        if with_vlffn:
            self.mlp_vl = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vl = norm_layer(dim)
        
        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        self.max_text_len = max_text_len

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias))

        if modality_type == "image":
            x = x + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x)))
        elif modality_type == "text":
            x = x + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x)))
        else:
            if self.mlp_vl is None:
                x_text = x[:, : self.max_text_len]
                x_imag = x[:, self.max_text_len :]
                x_text = x_text + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x_text)))
                x_imag = x_imag + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
                x = torch.cat([x_text, x_imag], dim=1)
            else:
                x = x + self.drop_path(self.gamma_2 * self.mlp_vl(self.norm2_vl(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        return x


class MultiWayTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        need_relative_position_embed=True,
        use_abs_pos_emb=False,
        layer_scale_init_values=0.1,
        vlffn_start_layer_index=10,
        max_text_len=40,
        config=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            need_relative_position_embed (bool): enable relative position bias on self-attention
            use_abs_pos_emb (bool): enable abs pos emb
            layer_scale_init_values (float or None): layer scale init values, set None to disable
            vlffn_start_layer_index (int): vl-ffn start index
            config: (dict): other hyper from pytorch-lighting
        """
        super().__init__()
        drop_path_rate = drop_path_rate if config is None else config["drop_path_rate"]
        # rank_zero_info("drop path rate: {}".format(drop_path_rate))
        self.use_abs_pos_emb = use_abs_pos_emb
        self.need_relative_position_embed = need_relative_position_embed

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.vlffn_start_layer_index = vlffn_start_layer_index

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if self.use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_vlffn=(i >= self.vlffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    max_text_len=max_text_len,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def visual_embed(self, _x):
        x = self.patch_embed(_x)
        x = x.flatten(2).transpose(1, 2)
        B, L, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        x_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)

        return x, x_mask
# VLMo small/p16 for cifar
@register_model
def vlmo_small_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 32)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=384, depth=7, num_heads=12, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=5, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# VLMo base/p16
@register_model
def vlmo_base_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=10, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# VLMo large/p16
@register_model
def vlmo_large_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=21, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# VLMo base+/p16
@register_model
def vlmo_base_plus_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=544, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=21,
        use_abs_pos_emb=True, need_relative_position_embed=False, 
        layer_scale_init_values=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def build_vlmo(args):

    v_num_class = args.v_class_num
    l_num_class = args.l_class_num

    if 'cifar100' in args.vision_data_dir:
        v_num_class = 100
    if 'agnews' in args.language_data_dir:
        l_num_class = 4

    model = VLMo(img_size=args.img_size,
                 patch_size=args.patch_size,
                 v_num_classes=v_num_class, 
                 l_num_classes=l_num_class, 
                 vocab_size=args.vocab_size, 
                 max_text_len=args.max_text_len,
                 drop_path_rate=args.drop_path_rate)

    return model

class VLMo(nn.Module):
    def __init__(self, img_size, patch_size=8, v_num_classes=100, l_num_classes=4, vocab_size=30522, max_text_len=40, drop_path_rate=0.1, config=None) -> None:
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        # self.transformer = create_model(
        #     model_str,
        #     img_size=self.img_size,
        #     pretrained=False,
        #     drop_rate=0,
        #     drop_path_rate=0,
        #     attn_drop_rate=0,
        #     drop_block_rate=None,
        #     config=config,
        # )

        self.transformer = MultiWayTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=384, depth=7, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=5, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate, config=config, max_text_len=max_text_len)

        self.patch_size = self.transformer.patch_size
        self.num_features = self.transformer.num_features
        self.num_layers = len(self.transformer.blocks)
        self.build_relative_position_embed()

        self.v_head = nn.Linear(self.num_features, v_num_classes)
        self.l_head = nn.Linear(self.num_features, l_num_classes)

        self.vlffn_start_layer_index = self.transformer.vlffn_start_layer_index

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=self.num_features,
            max_position_embeddings=max_text_len,
            hidden_dropout_prob=drop_path_rate,
            position_embedding_type="rel_pos" if self.transformer.need_relative_position_embed else "absolute", 
        )

        self.text_embeddings = BertEmbeddings(bert_config)

        self.token_type_embeddings = nn.Embedding(2, self.num_features)

    def build_relative_position_embed(self, max_text_len_of_initckpt=196, max_text_len=40):
        if not self.transformer.need_relative_position_embed:
            self.relative_position_embed = False
            self.text_imag_relative_position_index = None
            self.text_relative_position_index = None
            self.relative_position_index = None
            return
        self.relative_position_embed = True
        window_size = (int(self.img_size / self.patch_size), int(self.img_size / self.patch_size)) #(14, 14)
        # rank_zero_info("window_size: {}".format(window_size))
        num_heads = self.transformer.num_heads
        # max_text_len_of_initckpt = max_text_len_of_initckpt #196
        # max_text_len = config["max_text_len"] #40
        max_imag_len = window_size[0] * window_size[1] + 1 #197
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.text_num_relative_distance = 2 * max_text_len_of_initckpt
        self.all_num_relative_distance = self.num_relative_distance + self.text_num_relative_distance + 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.all_num_relative_distance, num_heads * self.num_layers))
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.relative_position_index = relative_position_index
        
        text_position_ids = torch.arange(max_text_len-1)
        text_rel_pos_mat = text_position_ids.unsqueeze(-2) - text_position_ids.unsqueeze(-1)
        min_distance = int(2-max_text_len_of_initckpt) #-194
        # rank_zero_info("min_distance: {}".format(min_distance))
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += (self.num_relative_distance + 2)
        text_relative_position_index = \
            torch.zeros(size=(max_text_len, ) * 2, dtype=relative_coords.dtype)
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        self.text_relative_position_index = text_relative_position_index
        
        text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len) * (self.num_relative_distance)
        imag2text_relative_position_index = torch.ones(max_imag_len, max_text_len) * (self.num_relative_distance + 1)

        text_row_relative_position_index = torch.cat((text_relative_position_index, text2imag_relative_position_index), 1)
        imag_row_relative_position_index = torch.cat((imag2text_relative_position_index, relative_position_index), 1)
        text_imag_relative_position_index = torch.cat((text_row_relative_position_index, imag_row_relative_position_index), 0)
        self.text_imag_relative_position_index = text_imag_relative_position_index


    def get_rel_pos_bias(self, relative_position_index):
        if self.relative_position_embed:
            relative_position_bias = F.embedding(relative_position_index.long().to(self.relative_position_bias_table.device),
                                                    self.relative_position_bias_table)
            all_relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, x, y
            relative_position_bias_list = torch.chunk(all_relative_position_bias, self.num_layers, dim=0)
            return relative_position_bias_list
        else:
            return [None] * self.num_layers

    def forward(self, x, modality, masks=None):

        if modality == 'v':
            image_embeds, image_masks = self.transformer.visual_embed(x)
            
            
            x = image_embeds
            all_hidden_states = []
            relative_position_bias_list = self.get_rel_pos_bias(self.relative_position_index)

            for i, blk in enumerate(self.transformer.blocks):
                x = blk(x, mask=image_masks, modality_type="image", relative_position_bias=relative_position_bias_list[i])
                all_hidden_states.append(x)
            
            # vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index-1]
            # for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            #     vlffn_hiddens = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=image_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[vlffn_index])
            
            vffn_hiddens = all_hidden_states[-1]

            vffn_hiddens = self.transformer.norm(vffn_hiddens)

            text_feats, image_feats = (
                None,
                vffn_hiddens,
            )

            cls_feats = self.v_head(image_feats[:, 0])
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

            # cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

            # vlffn_hiddens = self.transformer.norm(vlffn_hiddens)
            # cls_vlffn_feats = self.itc_vl_image_proj(vlffn_hiddens[:, 0])
            # cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

            return cls_feats
        elif modality == 'l':

            text_embeds = self.text_embeddings(x)
            co_masks = masks

            x = text_embeds
            all_hidden_states = []
            relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

            

            for i, blk in enumerate(self.transformer.blocks):
                x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
                all_hidden_states.append(x)
            
            # vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index-1]
            # for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            #     vlffn_hiddens = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[vlffn_index])

            lffn_hiddens = all_hidden_states[-1]

            lffn_hiddens = self.transformer.norm(lffn_hiddens)
            text_feats, image_feats = (
                lffn_hiddens,
                None,
            )

            cls_feats = self.l_head(text_feats[:, 0])
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

            return cls_feats
        
    def state_dict(self, modality='vl'):
        dicts = {}
        v_params = ['imag', 'patch_embed', 'pos_embed', 'v_head', 'cls_token']
        l_params = ['text', 'l_head']

        if modality == 'v':
            for k, p in self.named_parameters():
                if all([i not in k for i in l_params]):
                    dicts.update({k: p})
        elif modality == 'l':
            for k, p in self.named_parameters():
                # if 'imag' not in k and 'patch_embed' not in k and 'v_head' not in k and 'cls_token' not in k:
                if all([i not in k for i in v_params]):
                    dicts.update({k: p})
        elif modality == 'vl':
            return super().state_dict()
        else:
            raise 'Unsupported modality'
        
        return dicts
    
    def load_state_dict(self, state_dict, strict=False):
        return super().load_state_dict(state_dict, strict=False)








    # def infer(
    #     self,
    #     batch,
    #     mask_text=False,
    #     mask_image=False,
    #     image_token_type_idx=1,
    #     image_embeds=None,
    #     image_masks=None,
    # ):
    #     if f"image_{image_token_type_idx - 1}" in batch:
    #         imgkey = f"image_{image_token_type_idx - 1}"
    #     else:
    #         imgkey = "image"

    #     do_mlm = "_mlm" if mask_text else ""
    #     text_ids = batch[f"text_ids{do_mlm}"]
    #     text_labels = batch[f"text_labels{do_mlm}"]
    #     text_masks = batch[f"text_masks"]
    #     text_embeds = self.text_embeddings(text_ids)

    #     img = batch[imgkey][0]
    #     image_embeds, image_masks = self.transformer.visual_embed(img)

    #     image_masks = image_masks.long().to(device=img.get_device())
    #     text_embeds, image_embeds = (
    #         text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
    #         image_embeds
    #         + self.token_type_embeddings(
    #             torch.full_like(image_masks, image_token_type_idx)
    #         ),
    #     )

    #     co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
    #     co_masks = torch.cat([text_masks, image_masks], dim=1)

    #     x = co_embeds
    #     relative_position_bias_list = self.get_rel_pos_bias(self.text_imag_relative_position_index)

    #     for i, blk in enumerate(self.transformer.blocks):
    #         x = blk(x, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[i])

    #     x = self.transformer.norm(x)
    #     text_feats, image_feats = (
    #         x[:, : text_embeds.shape[1]],
    #         x[:, text_embeds.shape[1] :],
    #     )
    #     cls_feats = self.pooler(x)

    #     ret = {
    #         "text_feats": text_feats,
    #         "image_feats": image_feats,
    #         "cls_feats": cls_feats,
    #         "raw_cls_feats": x[:, 0],
    #         "image": img,
    #         "text_labels": text_labels,
    #         "text_ids": text_ids,
    #         "text_masks": text_masks,
    #     }

    #     return ret