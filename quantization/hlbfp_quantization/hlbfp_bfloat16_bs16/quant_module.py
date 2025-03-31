import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from para_config import *
from quant_function import BFPQuantFunction
from hlbfp_format import HybridLowBlockFP

class HMQAct(nn.Module):
    """Class to quantize given activations

    Args:
        act_format (HybridLowBlockFP): The bfp format for activations
    """
    def __init__(self, layer_format: list):
        super(HMQAct, self).__init__()
        self.act_format = layer_format[0]

    def forward(self, x):
        bfp_act = BFPQuantFunction.apply(x, self.act_format)
        return bfp_act

class HMQLinear(nn.Linear):
    """Class to quantize linear layers

    Args:
        act_format (BlockFloatingPoint): The bfp format for activations
        w_format (BlockFloatingPoint): The bfp format for weights
        bias_format (BlockFloatingPoint): The bfp format for bias
        in_features (int): The number of input features
        out_features (int): The number of output features
        bias (bool, optional): Whether to use bias. Defaults to True.
    """

    def __init__(
        self,
        layer_format: list,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super(HMQLinear, self).__init__(in_features, out_features, bias)
        self.act_format = layer_format[0]
        self.w_format = layer_format[1]
        self.bias_format = layer_format[2]
        self.if_use_bias = bias

    def forward(self, x, out_quant=True):
        bfp_w = BFPQuantFunction.apply(self.weight, self.w_format)
        if self.if_use_bias:

            bfp_bias = BFPQuantFunction.apply(self.bias, self.bias_format)
        else:
            bfp_bias = self.bias
        bfp_out = F.linear(x, bfp_w, bfp_bias)
        if out_quant:

            bfp_out = BFPQuantFunction.apply(bfp_out, self.act_format)
        return bfp_out

# class HMQConv2d(nn.Conv2d):
#     """Class to quantize convolutional layers"""

#     def __init__(
#         self,
#         layer_format: list,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         bias=True,
#     ):
#         super(HMQConv2d, self).__init__(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#         )
#         self.act_format = layer_format[0]
#         self.w_format = layer_format[1]
#         self.bias_format = layer_format[2]
#         self.if_use_bias = bias

#     def forward(self, x, out_quant=True):
#         bfp_w = BFPQuantFunction.apply(self.weight, self.w_format)
#         if self.if_use_bias:

#             bfp_bias = BFPQuantFunction.apply(self.bias, self.bias_format)
#         else:
#             bfp_bias = self.bias
#         bfp_out = F.conv2d(
#             x,
#             bfp_w,
#             bfp_bias,
#             self.stride,
#             self.padding,
#             self.dilation,
#             self.groups,
#         )
#         if out_quant:

#             bfp_out = BFPQuantFunction.apply(bfp_out, self.act_format)
#         return bfp_out

def im2col(img, kernel_h, kernel_w, stride=1):
    N, C, H, W = img.shape
    out_h = (H - kernel_h) // stride[0] + 1
    out_w = (W - kernel_w) // stride[1] + 1

    col = torch.zeros(N, C, kernel_h, kernel_w, out_h, out_w).to(img.device)
    for y in range(kernel_h):
        y_max = y + stride[1] * out_h
        for x in range(kernel_w):
            x_max = x + stride[0] * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride[1], x:x_max:stride[0]]
    col = col.permute(0, 4, 5, 1, 2, 3).contiguous().view(N * out_h * out_w, -1)
    return col

def my_conv(input, weight, bias=None, stride=1, padding=0):
    # if padding > 0:
    #     input = F.pad(input, (padding, padding, padding, padding))
    batch_size = input.shape[0]
    input_h, input_w = input.shape[2], input.shape[3]
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    out_channel, in_channel = weight.shape[0], weight.shape[1]
    out_h = math.floor((input_h - kernel_h) / stride[0] + 1)
    out_w = math.floor((input_w - kernel_w) / stride[1] + 1)

    input_vector = im2col(input, kernel_h, kernel_w, stride)
    weight_vector = weight.reshape(weight.shape[0], -1).T
    output_vector = input_vector @ weight_vector + bias

    output = output_vector.reshape(batch_size, out_h, out_w, out_channel).permute(0, 3, 1, 2)
    return output

class HMQConv2d(nn.Conv2d):
    """Class to quantize convolutional layers"""

    def __init__(
        self,
        layer_format: list,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(HMQConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.act_format = layer_format[0]
        self.w_format = layer_format[1]
        self.bias_format = layer_format[2]
        self.if_use_bias = bias

    def forward(self, x, out_quant=True):
        bfp_w = BFPQuantFunction.apply(self.weight, self.w_format)
        if self.if_use_bias:
            bfp_bias = BFPQuantFunction.apply(self.bias, self.bias_format)
        else:
            bfp_bias = self.bias

        # bfp_out = F.conv2d(
        #     x,
        #     bfp_w,
        #     bfp_bias,
        #     self.stride,
        #     self.padding,
        #     self.dilation,
        #     self.groups,
        # )
        conv_out = my_conv(x, bfp_w, bfp_bias, self.stride, self.padding)

        if out_quant:
            bfp_out = BFPQuantFunction.apply(conv_out, self.act_format)
        return bfp_out

class HMQMulQKT(nn.Module):
    def __init__(self, layer_format: list):
        super(HMQMulQKT, self).__init__()
        self.act_format = layer_format[0]

    def forward(self, q, k, out_quant=True):
        bfp_out = q @ k.transpose(-2, -1)
        if out_quant:

            bfp_out = BFPQuantFunction.apply(bfp_out, self.act_format)
        return bfp_out

class HMQMulSV(nn.Module):
    def __init__(self, layer_format: list):
        super(HMQMulSV, self).__init__()
        self.act_format = layer_format[0]

    def forward(self, s, v):
        bfp_out = s @ v
        bfp_out = BFPQuantFunction.apply(bfp_out, self.act_format)
        return bfp_out

def isqrt(x):
    y = torch.tensor(x).bfloat16()
    i = y.view(torch.short)
    i = torch.tensor(0x5f37).short() - torch.tensor(i >> 1).short()
    y = i.view(torch.bfloat16)
    
    y = torch.tensor(y).bfloat16()
    y_tmp = (y * y).bfloat16()
    x_tmp = torch.tensor(x * 0.5).bfloat16()

    # x_tmp = x_tmp * y
    # mul_tmp = (x_tmp * y).bfloat16()
    mul_tmp = (x_tmp * y_tmp).bfloat16()
    sub_tmp = (1.5 - mul_tmp).bfloat16()
    y = y * sub_tmp
    return y
    

def pade_tanh_tensor(x: torch.Tensor) -> torch.Tensor:
    """ a rational function to approximate a tanh-like soft clipper based on 
        the pade-approximation of the tanh function with tweaked coefficients.

        The function is in the range x=-3..3 and outputs the range y=-1..1. 
        Beyond this range the output must be clamped to -1..1.

        The first to derivatives of the function vanish at -3 and 3, 
        so the transition to the hard clipped region is C2-continuous.
    """
    numerator = x * ( 27 + x * x )
    denominator = ( 27 + 9 * x * x )

    mask = denominator <= 0
    denominator[mask] = -denominator[mask]
    numerator[mask] = -numerator[mask]
    den = isqrt(denominator)
    result = (numerator * den * den).bfloat16()

    tanh_x = torch.clamp(result, min=-1, max=1)
    return tanh_x


# def pade_tanh_tensor(x: torch.Tensor) -> torch.Tensor:
#     # Coefficients for a higher-order Padé approximant of tanh
#     # b0, b1, b2, b3 = (torch.tensor(1.1021)).bfloat16(), (torch.tensor(0.5541)).bfloat16(), (torch.tensor(-0.1119)).bfloat16(), (torch.tensor(-0.0075)).bfloat16(),
#     b0, b1, b2, b3 = (torch.tensor(1.1021)).bfloat16(), (torch.tensor(0.5 + 0.044715 + 0.0026092529296875)).bfloat16(), (torch.tensor(-0.1119)).bfloat16(), (torch.tensor(-0.0075)).bfloat16(),
#     # a0, a1, a2, a3 = (torch.tensor(1.1019)).bfloat16(), (torch.tensor(0.9233)).bfloat16(), (torch.tensor(0.044715)).bfloat16(), (torch.tensor(-0.0522)).bfloat16()
#     a0, a1, a2, a3 = (torch.tensor(1.1019)).bfloat16(), (torch.tensor(1-0.1119)).bfloat16(), (torch.tensor(0.044715)).bfloat16(), (torch.tensor(-0.0522)).bfloat16()

#     # Compute the numerator and denominator of the Padé approximant
#     x2 = (x * x).bfloat16()
#     x4 = (x2 * x2).bfloat16()
#     x6 = (x4 * x2).bfloat16()

#     num_mul1 = (b0 * x).bfloat16()
#     num_mul2 = ((b1 * x2).bfloat16() * x).bfloat16()
#     num_mul3 = ((b2 * x4).bfloat16() * x).bfloat16()
#     num_mul4 = ((b3 * x6).bfloat16() * x).bfloat16()

#     den_mul1 = (b0.clone().detach()).bfloat16() # a0
#     den_mul2 = (a1 * x2).bfloat16()
#     den_mul3 = (a2 * x4).bfloat16()
#     den_mul4 = (a3 * x6).bfloat16()
    
#     numerator = (((num_mul1 + num_mul2).bfloat16() + num_mul3).bfloat16() + num_mul4).bfloat16()
#     denominator = (((den_mul1 + den_mul2).bfloat16() + den_mul3).bfloat16() + den_mul4).bfloat16()

#     # The Padé approximant of tanh(x)
#     # result = numerator / denominator
#     mask = denominator <= 0
#     denominator[mask] = -denominator[mask]
#     numerator[mask] = -numerator[mask]
#     den = isqrt(denominator)
#     result = numerator * den * den

#     result  = result.bfloat16()
#     result = torch.clamp(result, min=-1, max=1)

#     return result

class HMQSoftmax(nn.Module):
    def __init__(self, layer_format:list):
        super(HMQSoftmax, self).__init__()
        self.sftm_act_format = HybridLowBlockFP(BLOCK_SIZE, 8, 7, 7)

    def forward(self, x):

        q = torch.floor( x / 0.6931471805599453 ).bfloat16()
        exp_x = (2 ** q).bfloat16()
        exp_x_sum = torch.sum(exp_x, dim=-1, keepdim=True).bfloat16()

        # x_tmp = torch.sum(x, dim=-1, keepdim=True).bfloat16()
        # q = torch.floor( x_tmp / 0.6931471805599453 ).bfloat16()
        # exp_x = (2 ** q).bfloat16()
        # exp_x_sum = exp_x

        exp_xs = isqrt(exp_x_sum)
        bfp_out = exp_x * exp_xs * exp_xs
        
        bfp_out = BFPQuantFunction.apply(bfp_out.float(), self.sftm_act_format) # TODO: convert bfloat16 to bfp

        return bfp_out

class HMQGeLU(nn.GELU):
    def __init__(self):
        super(HMQGeLU, self).__init__()
        self.m = nn.Sigmoid()
        self.gelu_act_format = HybridLowBlockFP(BLOCK_SIZE, 4, 7, 7)

    def forward(self, x):
        x = x.bfloat16()

        # tmp_x = x
        # pow_tmp = 0.044715*torch.pow(x, 3)
        # pow_tmp = pow_tmp.bfloat16()
        # pi_tmp = math.sqrt(2/math.pi)
        # pi_tmp = torch.tensor(pi_tmp).bfloat16()
        # add_tmp = (x + pow_tmp).bfloat16()
        # x = pi_tmp*add_tmp

        # x = pade_tanh_tensor(x) # Padé Approximation

        # # q = torch.floor( 2 * x * ( 1 / 0.6931471805599453) )
        # # q = q.bfloat16()
        # # exp_2x = 2 ** q
        # # x = (exp_2x - 1).bfloat16() / (exp_2x + 1).bfloat16() # tanh(x)

        # # x = (torch.exp(2*x) - 1) / (torch.exp(2*x) + 1)
        # # x = torch.sinh(x) / torch.cosh(x)
        # # x = torch.tanh(x)

        # x = x.bfloat16()
        # x += 1
        # mul_tmp = tmp_x * x
        # mul_tmp = mul_tmp.bfloat16()
        # bfp_out = 0.5 * mul_tmp
        # bfp_out = BFPQuantFunction.apply(bfp_out.float(), self.gelu_act_format) # TODO: convert bfloat16 to bfp
        

        x_tmp = (1.702 * x).bfloat16()
        # bfp_out = (self.m(x_tmp) * x).bfloat16()

        bfp_out = (x / (1 + torch.exp(-x_tmp).bfloat16()).bfloat16()).bfloat16()

        # q = torch.floor( -x_tmp / 0.6931471805599453 )
        # exp_x = 2 ** q
        # bfp_out = x / (1 + exp_x)
        

        bfp_out = BFPQuantFunction.apply(bfp_out.float(), self.gelu_act_format) # TODO: convert bfloat16 to bfp

        
        return bfp_out

class HMQLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(HMQLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.ln_act_format = HybridLowBlockFP(BLOCK_SIZE, 4, 7, 7)

    def forward(self, x):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        x = x.bfloat16()

        # mean = x.mean(dim=dims, keepdim=True).bfloat16()
        x_sum = torch.sum(x, dim=dims, keepdim=True).bfloat16()
        mean = (x_sum / x.shape[-1]).bfloat16()
 
        pow_tmp = (x ** 2).bfloat16()

        # mean_x2 = (pow_tmp).mean(dim=dims, keepdim=True).bfloat16()
        pow_tmp_sum = torch.sum(pow_tmp, dim=dims, keepdim=True).bfloat16()
        mean_x2 = (pow_tmp_sum / pow_tmp.shape[-1]).bfloat16()
 
        mean_pow_tmp = (mean ** 2).bfloat16()
        var = (mean_x2 - mean_pow_tmp).bfloat16()
        add_tmp = (var + torch.tensor(self.eps).bfloat16()).float()
        var_sqrt = isqrt(add_tmp.cpu().detach().numpy()).to('cuda').bfloat16() # Inverse Sqrt
        sub_tmp = (x - mean).bfloat16()
        x_norm = (sub_tmp * var_sqrt).bfloat16()

        if self.elementwise_affine:
            w_x_tmp = (self.weight * x_norm).bfloat16()
            x_norm = w_x_tmp + self.bias.bfloat16()

        bfp_out = x_norm
        bfp_out = BFPQuantFunction.apply(bfp_out.float(), self.ln_act_format) # TODO: convert bfloat16 to bfp

        return bfp_out
