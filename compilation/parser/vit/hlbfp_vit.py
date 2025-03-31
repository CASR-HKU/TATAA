import torch
from torch import nn
from collections import OrderedDict
from functools import partial
from hlbfp_sub_layers import *
from hlbfp_format import HybridLowBlockFP
from quant_module import HMQAct, HMQLinear, HMQLayerNorm
from para_config import *
from parser.model_parser import ModelParser

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

from time import perf_counter

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=8,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        norm_layer=None,
        model_format=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        embed_format_dict = model_format[0]
        attn_format_dicts = model_format[1]
        mlp_format_dicts = model_format[2]
        head_format_dict = model_format[3]

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            format_dict=embed_format_dict,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    attn_format_dict=attn_format_dicts[i],
                    mlp_format_dict=mlp_format_dicts[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        head_format = head_format_dict["head"]
        self.head = (
            HMQLinear(head_format, self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        input_format = HybridLowBlockFP(BLOCK_SIZE, 4, 7, 7)
        self.input_quant = HMQAct([input_format])

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x, mp: ModelParser, C_dict=None):
        x = self.patch_embed(x, mp)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x_posembed = torch.cat((cls_tokens, x), dim=1)
        # mp.parse_layer('position_embed', 'PosEmbed')
        # mp.parse_ops('conc', in_data=cls_tokens, in_data_name='A31', in_tmp=x, in_tmp_name='A0', out_data=x_posembed, out_data_name='B0', verbose=True)
        x = x_posembed + self.pos_embed
        # mp.parse_ops('add', in_data=x_posembed, in_data_name='B0', in_tmp=self.pos_embed.repeat(x_posembed.shape[0], 1, 1).flatten(0,1), in_tmp_name='A30', out_data=x, out_data_name='A0', verbose=True)

        for i, blk in enumerate(self.blocks):
            # start.record()
            start_time = perf_counter()
            x = blk(x, mp, i, C_dict)
            # end.record()
            # torch.cuda.synchronize()
            # print(start.elapsed_time(end))
            print("Power: ", torch.cuda.power_draw())
            torch.cuda.synchronize()
            end_time = perf_counter()
            print(end_time - start_time)
            break
        return x

    def forward(self, x, mp: ModelParser, C_dict=None):
        
        # x = self.input_quant(x)

        x = self.forward_features(x, mp, C_dict)
        
        # x = self.head(x)
        return x

def deit_tiny_patch16_224(pretrained=False, model_format=None, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(HMQLayerNorm, eps=1e-6),
        model_format=model_format,
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def deit_small_patch16_224(pretrained=False, model_format=None, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(HMQLayerNorm, eps=1e-6),
        model_format=model_format,
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def deit_base_patch16_224(pretrained=False, model_format=None, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(HMQLayerNorm, eps=1e-6),
        model_format=model_format,
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
