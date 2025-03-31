import torch
from torch import nn
from para_config import *
from parser.model_parser import ModelParser
from hlbfp_format import HybridLowBlockFP
from quant_module import (
    HMQAct,
    HMQLinear,
    HMQConv2d,
    HMQKV,
    HMQMulQKT,
    HMQMulSV,
    HMQLayerNorm,
    HMQSoftmax,
    HMQGeLU,
)

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=8,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        format_dict: dict = {"embed": None},
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.layer_format_proj = format_dict["embed"]
        if self.layer_format_proj == None:
            raise ValueError("format_dict is not set correctly")

        self.proj = HMQConv2d(
            self.layer_format_proj,
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, mp: ModelParser):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x, 'conv', mp)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        format_dict: dict = {
            "qkv": None,
            "mulqk": None,
            "sftm": None,
            "mulsv": None,
            "proj": None,
        },
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.layer_format_qkv = format_dict["qkv"]
        self.layer_format_mul_qk = format_dict["mulqk"]
        self.layer_format_sftm = format_dict["sftm"]
        self.layer_format_mul_sv = format_dict["mulsv"]
        self.layer_format_proj = format_dict["proj"]

        self.qkv = HMQKV(self.layer_format_qkv, dim, dim * 3, bias=qkv_bias)
        self.mul_qk = HMQMulQKT(self.layer_format_mul_qk)
        self.sftm = HMQSoftmax()
        self.mul_sv = HMQMulSV(self.layer_format_mul_sv)
        self.proj = HMQLinear(self.layer_format_proj, dim, dim)

    def forward(self, x, mp: ModelParser, i: int, C_dict=None):

        q, k, v = self.qkv(x, self.num_heads, 'block.'+str(i)+'.qkv', mp, node_type='qkv')
        attn = self.mul_qk(q, k, 'block.'+str(i)+'.mul_qk', mp, self.scale)
        attn = self.sftm(attn, 'block.'+str(i)+'.sftm', mp, C_dict)
        x = self.mul_sv(attn, v, x, 'block.'+str(i)+'.mul_sv', mp)
        x = self.proj(x, 'block.'+str(i)+'.proj', mp, quant_type='fp16')

        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        format_dict: dict = {"fc1": None, "fc2": None},
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layer_format_fc1 = format_dict["fc1"]
        self.layer_format_fc2 = format_dict["fc2"]

        self.fc1 = HMQLinear(self.layer_format_fc1, in_features, hidden_features)
        self.act = HMQGeLU()
        self.fc2 = HMQLinear(self.layer_format_fc2, hidden_features, out_features)

    def forward(self, x, mp: ModelParser, i: int, C_dict=None):
        x = self.fc1(x, 'block.'+str(i)+'.fc1', mp, quant_type='fp16')
        x = self.act(x, 'block.'+str(i)+'.gelu', mp, C_dict)
        x = self.fc2(x, 'block.'+str(i)+'.fc2', mp, quant_type='fp16')
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=HMQLayerNorm,
        attn_format_dict: dict = {
            "qkv": None,
            "mulqk": None,
            "sftm": None,
            "mulsv": None,
            "proj": None,
        },
        mlp_format_dict: dict = {"fc1": None, "fc2": None},
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm1_act_format = HybridLowBlockFP(BLOCK_SIZE, 4, 7, 7)
        self.norm1_act = HMQAct([self.norm1_act_format])
        self.norm2_act_format = HybridLowBlockFP(BLOCK_SIZE, 4, 7, 7)
        self.norm2_act = HMQAct([self.norm2_act_format])
        
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            format_dict=attn_format_dict,
        )
        
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            format_dict=mlp_format_dict,
        )

    def forward(self, x, mp: ModelParser, i: int, C_dict=None):

        x = x + self.attn(self.norm1_act(self.norm1(x, 'block.'+str(i)+'.attn_norm', mp, C_dict), 'block.'+str(i)+'.attn_in_act', mp), mp, i, C_dict)
        x = x + self.mlp(self.norm2_act(self.norm2(x, 'block.'+str(i)+'.mlp_norm', mp, C_dict), 'block.'+str(i)+'.mlp_in_act', mp), mp, i, C_dict)

        return x
