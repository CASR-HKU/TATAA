import torch
from torch import nn

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

from hlibf_sub_layer import *
from quant_module import HMQAct, HMQLinear, HMQConv2d, HMQLayerNorm, HMQSoftmax, HMQGeLU, HMQMulQK, HMQMulSV

from parser.model_parser import ModelParser
from time import perf_counter
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 quant=False,
                 calibrate=False,
                 input_quant=False,
                 cfg=None,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.input_quant = input_quant
        self.cfg = cfg
        if input_quant:
            self.HMQAct_input = HMQAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            quant=quant,
            calibrate=calibrate,
            cfg=cfg)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            self.HMQAct1 = HMQAct(quant=quant,
                              calibrate=calibrate,
                              bit_type=cfg.BIT_TYPE_A,
                              calibration_mode=cfg.CALIBRATION_MODE_LN,
                              observer_str=cfg.OBSERVER_LN,
                              quantizer_str=cfg.QUANTIZER_LN)
        else:
            self.absolute_pos_embed = None

        # build layers
        layers = []
        print(" --------------- num_layer: ", self.num_layers, " --------------- ")
        for i_layer in range(self.num_layers):
            layers += [
                BasicLayer(
                    dim=int(embed_dim * 2**i_layer),
                    input_resolution=(self.patch_grid[0] // (2**i_layer),
                                      self.patch_grid[1] // (2**i_layer)),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    quant=quant,
                    calibrate=calibrate,
                    cfg=cfg)
            ]
        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(self.num_features)
        self.HMQAct2 = HMQAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.HMQAct3 = HMQAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.head = HMQLinear(self.num_features,
                            num_classes,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W
                            ) if num_classes > 0 else nn.Identity()

        self.act_out = HMQAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = HMQLinear(self.num_features,
                            num_classes,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W
                            ) if num_classes > 0 else nn.Identity()

    def model_quant(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.quant = True

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.calibrate = False

    def forward_features(self, x):
        if self.input_quant:
            x = self.HMQAct_input(x)
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
            x = self.HMQAct1(x)
        for i, layer in enumerate(self.layers):
            

            batch_size = 1
            mp = ModelParser('swin_tiny', batch_size)
            C_dict = {}

            print(" ----> Layer: ", i)
            x = layer(x, mp, C_dict, i)

            ms = mp.return_ms()    
            ms.update_param('constants', C_dict)
            ms.save(f"./model_spec/swin_tiny_layer{i}_bs{batch_size}.json")

            # break
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


def swin_tiny_patch4_window7_224(pretrained=False,
                                 quant=False,
                                 calibrate=False,
                                 cfg=None,
                                 **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model = SwinTransformer(patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            norm_layer=HMQLayerNorm,
                            quant=quant,
                            calibrate=calibrate,
                            input_quant=True,
                            cfg=cfg,
                            **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def swin_small_patch4_window7_224(pretrained=False,
                                  quant=False,
                                  calibrate=False,
                                  cfg=None,
                                  **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model = SwinTransformer(patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            norm_layer=HMQLayerNorm,
                            quant=quant,
                            calibrate=calibrate,
                            input_quant=True,
                            cfg=cfg,
                            **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def swin_base_patch4_window7_224(pretrained=False,
                                 quant=False,
                                 calibrate=False,
                                 cfg=None,
                                 **kwargs):
    """ Swin-B @ 224x224, trained ImageNet-1k
    """
    model = SwinTransformer(patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            norm_layer=HMQLayerNorm,
                            quant=quant,
                            calibrate=calibrate,
                            input_quant=True,
                            cfg=cfg,
                            **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model
