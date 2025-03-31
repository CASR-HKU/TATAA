import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from para_config import *
from parser.model_parser import ModelParser
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

    def forward(self, x, layer_name='', mp=None):
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

    def forward(self, x, layer_name='', mp=None, out_quant=True, quant_type='int8', node_type='mlp'):
        mp.parse_layer(layer_name, node_type)
        bfp_w = BFPQuantFunction.apply(self.weight, self.w_format)
        if self.if_use_bias:
            bfp_bias = BFPQuantFunction.apply(self.bias, self.bias_format)
        else:
            bfp_bias = self.bias
        ln_out = F.linear(x, bfp_w, bfp_bias)
        mp.parse_ops('matmul', in_data=self.weight, in_data_name='', in_tmp=x, in_tmp_name='', out_data=ln_out, out_data_name='', feature_dict={'bias_shape': self.bias.shape, 'scale': 0.1, 'quant_type': quant_type}, int_ops=True)
        if out_quant:
            bfp_out = BFPQuantFunction.apply(ln_out, self.act_format)
        return bfp_out

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

def my_conv(input, weight, bias=None, stride=1, padding=0, mp=None):
    if padding[0] > 0:
        input = F.pad(input, (padding, padding, padding, padding))
    batch_size = input.shape[0]
    input_h, input_w = input.shape[2], input.shape[3]
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    out_channel, in_channel = weight.shape[0], weight.shape[1]
    out_h = math.floor((input_h - kernel_h) / stride[0] + 1)
    out_w = math.floor((input_w - kernel_w) / stride[1] + 1)

    input_vector = im2col(input, kernel_h, kernel_w, stride)
    weight_vector = weight.reshape(weight.shape[0], -1).T
    output_vector = input_vector @ weight_vector + bias

    mp.parse_ops('matmul', in_data=torch.transpose(weight_vector, 0, 1), in_data_name='', 
                         in_tmp=torch.transpose(input_vector, 0, 1), in_tmp_name='', 
                         out_data=torch.transpose(output_vector, 0, 1), out_data_name='', 
                         feature_dict={'bias_shape': bias.shape, 'scale': 0.1, 'quant_type': 'fp16'}, int_ops=True)

    output = output_vector.reshape(batch_size, out_h, out_w, out_channel).permute(0, 3, 1, 2)
    return output, output_vector

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

    def forward(self, x, layer_name='', mp=None):
        mp.parse_layer(layer_name, 'mlp')
        bfp_w = BFPQuantFunction.apply(self.weight, self.w_format)
        if self.if_use_bias:
            bfp_bias = BFPQuantFunction.apply(self.bias, self.bias_format)
        else:
            bfp_bias = self.bias

        conv_out, out_vector = my_conv(x, bfp_w, bfp_bias, self.stride, self.padding, mp)
        bfp_out = BFPQuantFunction.apply(conv_out, self.act_format)

        return bfp_out


class HMQKV(nn.Linear):

    def __init__(
        self,
        layer_format: list,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super(HMQKV, self).__init__(in_features, out_features, bias)
        self.act_format = layer_format[0]
        self.w_format = layer_format[1]
        self.bias_format = layer_format[2]
        self.if_use_bias = bias

    def forward(self, x, num_heads, layer_name='', mp=None, out_quant=True, quant_type='int8', node_type='qkv'):
        mp.parse_layer(layer_name, node_type)
        B, N, C = x.shape
        bfp_w = BFPQuantFunction.apply(self.weight, self.w_format)
        if self.if_use_bias:
            bfp_bias = BFPQuantFunction.apply(self.bias, self.bias_format)
        else:
            bfp_bias = self.bias
        ln_out = F.linear(x, bfp_w, bfp_bias)

        bfp_out = BFPQuantFunction.apply(ln_out, self.act_format)

        x_reshape = bfp_out.reshape(B, N, 3, num_heads, C // num_heads)     
        qkv = x_reshape.permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = qkv[0], qkv[1], qkv[2] # q: A0, k: A1, v: A2
        # mp.parse_ops('matmul', in_data=self.weight, in_data_name='', in_tmp=x.transpose(1, 2), in_tmp_name='', out_data=ln_out.transpose(1,2), out_data_name='', feature_dict={'bias_shape': self.bias.shape, 'scale': 0.1, 'quant_type': quant_type}, int_ops=True)
        mp.parse_ops('matmul', in_data=self.weight, in_data_name='', in_tmp=x, in_tmp_name='', out_data=ln_out, out_data_name='', feature_dict={'bias_shape': self.bias.shape, 'scale': 0.1, 'quant_type': quant_type}, int_ops=True)

        return q, k, v

class HMQMulQKT(nn.Module):
    def __init__(self, layer_format: list):
        super(HMQMulQKT, self).__init__()
        self.act_format = layer_format[0]

    def forward(self, q, k, layer_name='', mp=None, scale=0.0):
        mp.parse_layer(layer_name, 'kq_mul')

        k_trans = k.transpose(-2, -1)
        qk_out = q @ k_trans
        print("q shape: ", q.shape)
        print("k_trans shape: ", k_trans.shape)
        print("qk_out shape: ", qk_out.shape)
        # mp.parse_ops('matmul', in_data=torch.transpose(k_trans, 2, 3), in_data_name='', in_tmp=torch.transpose(q, 2, 3), in_tmp_name='', out_data=torch.transpose(qk_out, 2, 3), out_data_name='', feature_dict={'scale': scale, 'quant_type': 'fp16'}, int_ops=True, keep_out=True)
        # bfp_out = BFPQuantFunction.apply(qk_out, self.act_format)
        mp.parse_ops('matmul', in_data=k_trans, in_data_name='', in_tmp=q, in_tmp_name='', out_data=qk_out, out_data_name='', feature_dict={'scale': scale, 'quant_type': 'fp16'}, int_ops=True, keep_out=True)
        bfp_out = BFPQuantFunction.apply(qk_out, self.act_format)
        return bfp_out * scale

class HMQMulSV(nn.Module):
    def __init__(self, layer_format: list):
        super(HMQMulSV, self).__init__()
        self.act_format = layer_format[0]

    def forward(self, s, v, x, layer_name='', mp=None):
        mp.parse_layer(layer_name, 'vs_mul')
        B, N, C = x.shape

        sv_out = s @ v
        x = BFPQuantFunction.apply(sv_out, self.act_format)

        x_trans = x.transpose(1, 2)
        x = x_trans.reshape(B, N, C)

        # mp.parse_ops('matmul', in_data=torch.transpose(v, 2, 3), in_data_name='', in_tmp=torch.transpose(s, 2, 3), in_tmp_name='', out_data=torch.transpose(sv_out, 2, 3)
        # , out_data_name='', feature_dict={'scale': 0.1, 'hw_out': x.flatten(0,1).shape, 'quant_type': 'int8'}, int_ops=True)
        mp.parse_ops('matmul', in_data=v, in_data_name='', in_tmp=s, in_tmp_name='', out_data=sv_out
        , out_data_name='', feature_dict={'scale': 0.1, 'hw_out': x.flatten(0,1).shape, 'quant_type': 'int8'}, int_ops=True)

        return x


def isqrt_ln(x, mp: ModelParser, C_dict=None):
    
    y = x.bfloat16()
    i = y.view(torch.short)
    i = torch.tensor(0x5f37).short() - (i >> 1).short()
    
    # mag number: value embedded in each PE (no need to load from RF); Input loaded from Y0
    mp.parse_ops('mag', in_data=y, in_data_name='A0', out_data=i, out_data_name='A1', verbose=True)

    x_tmp = (x * 0.5).bfloat16()
    mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C2', out_data=x_tmp, out_data_name='A0', verbose=True)
    C_dict['C2'] = 0.5

    y = i.view(torch.bfloat16)
    y_tmp = y * y

    mp.parse_ops('mul', in_data=y, in_data_name='A1', in_tmp=y, in_tmp_name='A1', out_data=y_tmp, out_data_name='B1', verbose=True)
    y_tmp = y_tmp.bfloat16()

    mul_tmp = x_tmp * y_tmp
    mp.parse_ops('mul', in_data=x_tmp, in_data_name='A0', in_tmp=y_tmp, in_tmp_name='B1', out_data=mul_tmp, out_data_name='B1', verbose=True)
    mul_tmp = mul_tmp.bfloat16()
    
    sub_tmp = 1.5 - mul_tmp
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C3', in_tmp=mul_tmp, in_tmp_name='B1', out_data=sub_tmp, out_data_name='B1', feature_dict={"sub":"Y1"}, verbose=True)
    C_dict['C3'] = 1.5

    sub_tmp = sub_tmp.bfloat16()
    y_out = y * sub_tmp
    mp.parse_ops('mul', in_data=y, in_data_name='A1', in_tmp=sub_tmp, in_tmp_name='B1', out_data=y_out, out_data_name='B1', verbose=True)
    return y_out

def isqrt_sftm(x, mp: ModelParser, C_dict=None):
    
    y = x.bfloat16()
    i = y.view(torch.short)
    i = torch.tensor(0x5f37).short() - (i >> 1).short()
    
    verbose=False
    mp.parse_ops('mag', in_data=y, in_data_name="A0", out_data=i, out_data_name='B0', verbose=verbose)

    x_tmp = x * 0.5
    mp.parse_ops('mul', in_data=x, in_data_name="A0", in_tmp=torch.tensor([]), in_tmp_name='C2', out_data=x_tmp, out_data_name='A0', verbose=verbose)

    y = i.view(torch.bfloat16)

    tmp = x_tmp * y
    mul_tmp = tmp * y
    mp.parse_ops('mul', in_data=x_tmp, in_data_name='A0', in_tmp=y, in_tmp_name='B0', out_data=tmp, out_data_name='A0', verbose=verbose)
    mp.parse_ops('mul', in_data=tmp, in_data_name='A0', in_tmp=y, in_tmp_name='B0', out_data=mul_tmp, out_data_name='A0', verbose=verbose)
    
    sub_tmp = 1.5 - mul_tmp
    mp.parse_ops('add', in_data=mul_tmp, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C3', out_data=sub_tmp, out_data_name='A1', feature_dict={"sub":"Y0"}, verbose=verbose)

    y_out = y * sub_tmp
    mp.parse_ops('mul', in_data=sub_tmp, in_data_name='A0', in_tmp=y, in_tmp_name='B0', out_data=y_out, out_data_name='B0', verbose=verbose)
    return y_out

def isqrt_gelu(x, mp: ModelParser, inter_name='', layer_type='', C_dict=None):
    
    y = x.bfloat16()
    i = y.view(torch.short)
    i = torch.tensor(0x5f37).short() - (i >> 1).short()
    
    mp.parse_ops('mag', in_data=y, in_data_name='A1', out_data=i, out_data_name='B1', verbose=True)

    x_tmp = x * 0.5
    mp.parse_ops('mul', in_data=x, in_data_name='A1', in_tmp=torch.tensor([]), in_tmp_name='C2', out_data=x_tmp, out_data_name='A1', verbose=True)

    y = i.view(torch.bfloat16)

    mul_x_tmp = x_tmp * y
    mul_tmp = mul_x_tmp * y
    mp.parse_ops('mul', in_data=x_tmp, in_data_name='A1', in_tmp=y, in_tmp_name='B1', out_data=mul_x_tmp, out_data_name='A1', verbose=True)
    mp.parse_ops('mul', in_data=mul_x_tmp, in_data_name='A1', in_tmp=y, in_tmp_name='B1', out_data=mul_tmp, out_data_name='A1', verbose=True)

    sub_tmp = 1.5 - mul_tmp
    mp.parse_ops('add', in_data=mul_tmp, in_data_name='A1', in_tmp=torch.tensor([]), in_tmp_name='C3', out_data=sub_tmp, out_data_name='A1', feature_dict={"sub":"Y0"}, verbose=True)

    y_out = y * sub_tmp
    mp.parse_ops('mul', in_data=sub_tmp, in_data_name='A1', in_tmp=y, in_tmp_name='B1', out_data=y_out, out_data_name='A1', verbose=True)
    return y_out

def pade_tanh_tensor(x: torch.Tensor, mp: ModelParser, C_dict=None) -> torch.Tensor:

    x2 = x * x
    mp.parse_ops('mul', in_data=x, in_data_name='B0', in_tmp=x, in_tmp_name='B0', out_data=x2, out_data_name='B1', verbose=True)

    num_add = 27 + x2
    C_dict['C7'] = torch.tensor(9).bfloat16().item()
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=x2, in_tmp_name='B1', out_data=x2, out_data_name='B1', verbose=True)
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=x2, in_tmp_name='B1', out_data=x2, out_data_name='B1', verbose=True)
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=x2, in_tmp_name='B1', out_data=num_add, out_data_name='A1', verbose=True)
    numerator = x * num_add
    mp.parse_ops('mul', in_data=num_add, in_data_name='A1', in_tmp=x, in_tmp_name='B0', out_data=numerator, out_data_name='B0', verbose=True)

    den_mul = 9 * x2
    mp.parse_ops('mul', in_data=torch.tensor([]), in_data_name='C7', in_tmp=x2, in_tmp_name='B1', out_data=den_mul, out_data_name='B1', verbose=True)
    denominator = 27 + den_mul
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=den_mul, in_tmp_name='B1', out_data=den_mul, out_data_name='B1', verbose=True)
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=den_mul, in_tmp_name='B1', out_data=den_mul, out_data_name='B1', verbose=True)
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=den_mul, in_tmp_name='B1', out_data=denominator, out_data_name='A1', verbose=True)

    mask = denominator <= 0
    denominator[mask] = -denominator[mask]
    numerator[mask] = -numerator[mask]
    den = isqrt_gelu(denominator, mp, 'A1', 'gelu', C_dict)

    tmp = numerator * den
    result = tmp * den
    mp.parse_ops('mul', in_data=den, in_data_name='A1', in_tmp=numerator, in_tmp_name='B0', out_data=tmp, out_data_name='B0', verbose=True)
    mp.parse_ops('mul', in_data=den, in_data_name='A1', in_tmp=tmp, in_tmp_name='B0', out_data=result, out_data_name='B0', feature_dict={'constraint': 'clamp'}, verbose=True)

    tanh_x = torch.clamp(result, min=-1, max=1)
    return tanh_x


class HMQSoftmax(nn.Module):
    def __init__(self):
        super(HMQSoftmax, self).__init__()
        self.sftm_act_format = HybridLowBlockFP(BLOCK_SIZE, 8, 7, 7)

    def forward(self, x, layer_name='', mp=None, C_dict=None):
        
        mp.parse_layer(layer_name, 'sftm')
        q = torch.floor( x / 0.6931471805599453 )
        exp_x = 2 ** q
        exp_x_sum = torch.sum(exp_x, dim=-1, keepdim=True)

        # mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='B31', out_data=q, out_data_name='A0', feature_dict={'constraint':['floor', 'rsh']})
        # mp.parse_ops('acc', in_data=exp_x, in_data_name='A0', in_tmp_name='B0', out_data=exp_x_sum, out_data_name='A0')
        mp.parse_ops('mua', in_data=x, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='B0', out_data=exp_x_sum, out_data_name='A0', feature_dict={'constant': 'C4', 'constraint':['floor', 'rsh']})
        C_dict['C4'] = torch.tensor(1/0.6931471805599453).bfloat16().item()

        exp_xs = isqrt_sftm(exp_x_sum, mp, C_dict)
        tmp = exp_x * exp_xs
        div_out = tmp * exp_xs

        # mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='B31', out_data=q, out_data_name='A0', feature_dict={'constraint':['floor', 'rsh']})
        # mp.parse_ops('mul', in_data=exp_x, in_data_name='A0', in_tmp=exp_xs, in_tmp_name='B0', out_data=tmp, out_data_name='A0')
        # mp.parse_ops('mul', in_data=tmp, in_data_name='A0', in_tmp=exp_xs, in_tmp_name='B0', out_data=div_out, out_data_name='A0', feature_dict={'quant_type': 'int8'})
        mp.parse_ops('mmm', in_data=x, in_data_name='A0', out_data=div_out, out_data_name='A0', feature_dict={'y0_ref': 'C4', 'y1_ref': 'B0', 'y2_ref': 'B0', 'constraint':['floor', 'rsh'], 'quant_type': 'int8'})
        bfp_out = BFPQuantFunction.apply(div_out.float(), self.sftm_act_format)

        return bfp_out

class HMQGeLU(nn.GELU):
    def __init__(self):
        super(HMQGeLU, self).__init__()
        self.gelu_act_format = HybridLowBlockFP(BLOCK_SIZE, 4, 7, 7)

    def forward(self, x, layer_name='', mp=None, C_dict=None):
        mp.parse_layer(layer_name, 'gelu')
        x = x.bfloat16()
        x2 = x * x
        pow_out = x2 * x
        mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=x, in_tmp_name='A0', out_data=x2, out_data_name='B0', verbose=True)
        mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=x2, in_tmp_name='B0', out_data=pow_out, out_data_name='B0', verbose=True)
        
        pow_tmp = 0.044715*pow_out
        mp.parse_ops('mul', in_data=torch.tensor([]), in_data_name='C5', in_tmp=pow_out, in_tmp_name='B0', out_data=pow_tmp, out_data_name='B0', verbose=True)
        C_dict['C5'] = torch.tensor(0.044715).bfloat16().item()
        
        add_tmp = (x + pow_tmp).bfloat16()
        mp.parse_ops('add', in_data=x, in_data_name='A0', in_tmp=pow_tmp, in_tmp_name='B0', out_data=add_tmp, out_data_name='A1', verbose=True)
        
        pi_tmp = math.sqrt(2/math.pi)
        pi_tmp = torch.tensor(pi_tmp).bfloat16()
        trans_tmp = pi_tmp*add_tmp
        mp.parse_ops('mul', in_data=add_tmp, in_data_name='A1', in_tmp=torch.tensor([]), in_tmp_name='C6', out_data=trans_tmp, out_data_name='B0', verbose=True)
        C_dict['C6'] = torch.tensor(pi_tmp.item()).bfloat16().item()

        trans_tmp = pade_tanh_tensor(trans_tmp, mp, C_dict) # Padé Approximation

        trans_x_tmp = trans_tmp.bfloat16()
        out_tmp = trans_x_tmp + 1
        mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C2', in_tmp=trans_x_tmp, in_tmp_name='B0', out_data=trans_x_tmp, out_data_name='B0', verbose=True)
        mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C2', in_tmp=trans_x_tmp, in_tmp_name='B0', out_data=out_tmp, out_data_name='B0', verbose=True)

        mul_tmp = x * out_tmp
        mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=out_tmp, in_tmp_name='B0', out_data=mul_tmp, out_data_name='A0', verbose=True)

        mul_tmp = mul_tmp.bfloat16()
        mul_out = 0.5 * mul_tmp
        mp.parse_ops('mul', in_data=mul_tmp, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C2', out_data=mul_out, out_data_name='A0', verbose=True, feature_dict={'quant_type': 'int8'})

        bfp_out = BFPQuantFunction.apply(mul_out.float(), self.gelu_act_format)
        return bfp_out

class HMQLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(HMQLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.ln_act_format = HybridLowBlockFP(BLOCK_SIZE, 4, 7, 7)

    def forward(self, x, layer_name='', mp=None, C_dict=None):
        mp.parse_layer(layer_name, 'ln')
        
        x = x.bfloat16()
        
        x_sum = torch.sum(x, dim=-1, keepdim=True)
        mean = x_sum / x.shape[-1]

        C_dict['C0'] = torch.tensor(1/x.shape[-1]).bfloat16().item()
        pow_tmp = (x ** 2).bfloat16()
        pow_sum = torch.sum(pow_tmp, dim=-1, keepdim=True)
        mean_x2 = pow_sum / pow_tmp.shape[-1]        
        mp.parse_ops('ama', in_data=x, in_data_name='A0', in_tmp_name='B0', out_data=x_sum, out_data_name='A0', out_tmp_name='A1', verbose=True)
        mp.parse_ops('mul', in_data=x_sum, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C0', out_data=mean, out_data_name='B0', verbose=True)
        mp.parse_ops('mul', in_data=pow_sum, in_data_name='A1', in_tmp=torch.tensor([]), in_tmp_name='C0', out_data=mean_x2, out_data_name='A0', verbose=True)

        mean_pow_tmp = (mean ** 2).bfloat16()
        mp.parse_ops('mul', in_data=mean, in_data_name='B0', in_tmp=mean, in_tmp_name='B0', out_data=mean_pow_tmp, out_data_name='B1', verbose=True)

        var = (mean_x2 - mean_pow_tmp).bfloat16()
        mp.parse_ops('add', in_data=mean_x2, in_data_name='A0', in_tmp=mean_pow_tmp, in_tmp_name='B1', out_data=var, out_data_name='A0', feature_dict={"sub": "Y1"}, verbose=True)

        add_tmp = (var + torch.tensor(self.eps).bfloat16())
        mp.parse_ops('add', in_data=var, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C1', out_data=add_tmp, out_data_name='A0', verbose=True)
        C_dict['C1'] = torch.tensor(self.eps).bfloat16().item()

        var_sqrt = isqrt_ln(add_tmp, mp, C_dict).to('cuda').bfloat16() # Inverse Sqrt

        sub_tmp = (x - mean).bfloat16()
        x_norm = (sub_tmp * var_sqrt).bfloat16()
        if self.elementwise_affine:
            w_x_tmp = (self.weight * x_norm).bfloat16()            
            x_norm = w_x_tmp + self.bias.bfloat16()

        # mp.parse_ops('add', in_data=x, in_data_name='A0', in_tmp=mean, in_tmp_name='B0', out_data=sub_tmp, out_data_name='A0', feature_dict={"sub": "Y1"}, verbose=True)
        # mp.parse_ops('mul', in_data=sub_tmp, in_data_name='A0', in_tmp=var_sqrt, in_tmp_name='A2', out_data=x_norm, out_data_name='A0', verbose=True)
        # mp.parse_ops('mul', in_data=x_norm, in_data_name='A0', in_tmp=self.weight, in_tmp_name="BW", out_data=w_x_tmp, out_data_name='A0', verbose=True, feature_dict={'elementwise_affine': 'B29-31'})
        # mp.parse_ops('add', in_data=w_x_tmp, in_data_name='A0', in_tmp=self.bias, in_tmp_name="BB", out_data=x_norm, out_data_name='A0', verbose=True, feature_dict={'elementwise_affine': 'A29-31', 'quant_type': 'int8'})
        
        mean_shape = mean.transpose(1, 2).shape
        var_sqrt_shape = var_sqrt.transpose(1, 2).shape

        mp.parse_ops('amd', in_data=x, in_data_name='A0', out_data=x_norm, out_data_name='A0', verbose=True, feature_dict={'y0': mean_shape[1:], 'y1': var_sqrt_shape[1:], 'y2': self.weight.shape, 'y3': self.bias.shape, 'yrf0': 'B0', 'yrf1': 'B1', 'yrf2': 'BW', 'yrf3': 'BB', 'quant_type': 'int8'})

        nm_out = x_norm
        bfp_out = BFPQuantFunction.apply(nm_out.float(), self.ln_act_format) # TODO: convert bfloat16 to bfp
        return bfp_out
