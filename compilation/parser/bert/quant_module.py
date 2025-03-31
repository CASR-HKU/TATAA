import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from hibf_config import BitType
from build_observer import *
from build_quantizer import *


def pade_tanh_tensor(x: torch.Tensor, mp=None, C_dict=None) -> torch.Tensor:
    numerator = x * (27 + x * x)
    x2 = x * x
    num_add = 27 + x2

    mp.parse_ops('mul', in_data=x, in_data_name='B0', in_tmp=x, in_tmp_name='B0', out_data=x * x, out_data_name='B1', verbose=True)
    C_dict['C7'] = torch.tensor(9).bfloat16().item()
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=x2, in_tmp_name='B1', out_data=x2, out_data_name='B1', verbose=True)
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=x2, in_tmp_name='B1', out_data=x2, out_data_name='B1', verbose=True)
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=x2, in_tmp_name='B1', out_data=num_add, out_data_name='A1', verbose=True)
    mp.parse_ops('mul', in_data=num_add, in_data_name='A1', in_tmp=x, in_tmp_name='B0', out_data=numerator, out_data_name='B0', verbose=True)
    

    denominator = 27 + 9 * x * x

    den_mul = 9 * x2
    mp.parse_ops('mul', in_data=torch.tensor([]), in_data_name='C7', in_tmp=x2, in_tmp_name='B1', out_data=den_mul, out_data_name='B1', verbose=True)
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=den_mul, in_tmp_name='B1', out_data=den_mul, out_data_name='B1', verbose=True)
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=den_mul, in_tmp_name='B1', out_data=den_mul, out_data_name='B1', verbose=True)
    mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C7', in_tmp=den_mul, in_tmp_name='B1', out_data=denominator, out_data_name='A1', verbose=True)


    mask = denominator <= 0
    denominator[mask] = -denominator[mask]
    numerator[mask] = -numerator[mask]
    den = isqrt_gelu(denominator, mp)
    result = (numerator * den * den).bfloat16()

    tmp = numerator * den
    result = tmp * den
    mp.parse_ops('mul', in_data=den, in_data_name='A1', in_tmp=numerator, in_tmp_name='B0', out_data=tmp, out_data_name='B0', verbose=True)
    mp.parse_ops('mul', in_data=den, in_data_name='A1', in_tmp=tmp, in_tmp_name='B0', out_data=result, out_data_name='B0', feature_dict={'constraint': 'clamp'}, verbose=True)


    tanh_x = torch.clamp(result, min=-1, max=1)
    return tanh_x


def isqrt_ln(x, mp=None, C_dict=None):
    
    y = x.bfloat16()
    i = y.view(torch.short)
    i = torch.tensor(0x5f37).short() - (i >> 1).short()
    
    # mag number: value embedded in each PE (no need to load from RF); Input loaded from Y0

    x_tmp = (x * 0.5).bfloat16()
    
    

    y = i.view(torch.bfloat16)
    y_tmp = y * y

    
    y_tmp = y_tmp.bfloat16()

    mul_tmp = x_tmp * y_tmp
    
    mul_tmp = mul_tmp.bfloat16()
    
    sub_tmp = 1.5 - mul_tmp
    
    # C_dict['C3'] = 1.5

    sub_tmp = sub_tmp.bfloat16()
    y_out = y * sub_tmp
    
    if mp is not None:
        C_dict['C6'] = 0.5
        mp.parse_ops('mag', in_data=y, in_data_name='A0', out_data=i, out_data_name='A1', verbose=True)
        mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C6', out_data=x_tmp, out_data_name='A0', verbose=True)
        mp.parse_ops('mul', in_data=y, in_data_name='A1', in_tmp=y, in_tmp_name='A1', out_data=y_tmp, out_data_name='B1', verbose=True)
        mp.parse_ops('mul', in_data=x_tmp, in_data_name='A0', in_tmp=y_tmp, in_tmp_name='B1', out_data=mul_tmp, out_data_name='B1', verbose=True)
        mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C6', in_tmp=mul_tmp, in_tmp_name='B1', out_data=sub_tmp, out_data_name='B1', feature_dict={"sub":"Y1"}, verbose=True)
        mp.parse_ops('mul', in_data=y, in_data_name='A1', in_tmp=sub_tmp, in_tmp_name='B1', out_data=y_out, out_data_name='B1', verbose=True)

    return y_out

def isqrt_sftm(x, mp=None):
    
    y = x.bfloat16()
    i = y.view(torch.short)
    i = torch.tensor(0x5f37).short() - (i >> 1).short()
    
    verbose=False
    

    x_tmp = x * 0.5

    y = i.view(torch.bfloat16)

    tmp = x_tmp * y
    mul_tmp = tmp * y
    
    sub_tmp = 1.5 - mul_tmp

    y_out = y * sub_tmp
    if mp is not None:
        mp.parse_ops('mag', in_data=y, in_data_name="A0", out_data=i, out_data_name='B0', verbose=verbose)
        mp.parse_ops('mul', in_data=x, in_data_name="A0", in_tmp=torch.tensor([]), in_tmp_name='C2', out_data=x_tmp, out_data_name='A0', verbose=verbose)
        mp.parse_ops('mul', in_data=x_tmp, in_data_name='A0', in_tmp=y, in_tmp_name='B0', out_data=tmp, out_data_name='A0', verbose=verbose)
        mp.parse_ops('mul', in_data=tmp, in_data_name='A0', in_tmp=y, in_tmp_name='B0', out_data=mul_tmp, out_data_name='A0', verbose=verbose)
        mp.parse_ops('add', in_data=mul_tmp, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C3', out_data=sub_tmp, out_data_name='A1', feature_dict={"sub":"Y0"}, verbose=verbose)
        mp.parse_ops('mul', in_data=sub_tmp, in_data_name='A0', in_tmp=y, in_tmp_name='B0', out_data=y_out, out_data_name='B0', verbose=verbose)
    
    return y_out

def isqrt_gelu(x, mp=None):
    
    y = x.bfloat16()
    i = y.view(torch.short)
    i = torch.tensor(0x5f37).short() - (i >> 1).short()
    
    

    x_tmp = x * 0.5
    
    y = i.view(torch.bfloat16)

    mul_x_tmp = x_tmp * y
    mul_tmp = mul_x_tmp * y

    sub_tmp = 1.5 - mul_tmp

    y_out = y * sub_tmp
    if mp is not None:
        mp.parse_ops('mag', in_data=y, in_data_name='A1', out_data=i, out_data_name='B1', verbose=True)
        mp.parse_ops('mul', in_data=x, in_data_name='A1', in_tmp=torch.tensor([]), in_tmp_name='C2', out_data=x_tmp, out_data_name='A1', verbose=True)

        mp.parse_ops('mul', in_data=x_tmp, in_data_name='A1', in_tmp=y, in_tmp_name='B1', out_data=mul_x_tmp, out_data_name='A1', verbose=True)
        mp.parse_ops('mul', in_data=mul_x_tmp, in_data_name='A1', in_tmp=y, in_tmp_name='B1', out_data=mul_tmp, out_data_name='A1', verbose=True)

        mp.parse_ops('add', in_data=mul_tmp, in_data_name='A1', in_tmp=torch.tensor([]), in_tmp_name='C3', out_data=sub_tmp, out_data_name='A1', feature_dict={"sub":"Y0"}, verbose=True)

        mp.parse_ops('mul', in_data=sub_tmp, in_data_name='A1', in_tmp=y, in_tmp_name='B1', out_data=y_out, out_data_name='A1', verbose=True)
    return y_out



# def isqrt(x):
#     y = x.bfloat16()
#     i = y.view(torch.short)
#     i = torch.tensor(0x5F37).short() - (i >> 1).short()
#     y = i.view(torch.bfloat16)

#     y_tmp = (y * y).bfloat16()
#     x = (x * 0.5).bfloat16()

#     x_tmp = x.bfloat16()
#     mul_tmp = x_tmp * y_tmp
#     mul_tmp = mul_tmp.bfloat16()
#     sub_tmp = 1.5 - mul_tmp
#     sub_tmp = sub_tmp.bfloat16()
#     y = y * sub_tmp
#     return y


class HMQConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        quant=False,
        calibrate=False,
        last_calibrate=False,
        bit_type=BitType,
        calibration_mode="layer_wise",
        observer_str="minmax",
        quantizer_str="uniform",
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
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "conv_weight"
        self.observer = build_observer(
            self.module_type, self.bit_type, self.calibration_mode
        )
        self.quantizer = build_quantizer(self.bit_type, self.observer, self.module_type)

    def forward(self, x, x_quantizer=None):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params()
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        weight = self.quantizer(self.weight, False)
        bias = self.bias / (self.quantizer.scale * x_quantizer.scale)
        x = F.conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        x = x * x_quantizer.scale * self.quantizer.scale

        return x


class HMQLinear(nn.Linear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        quant=False,
        calibrate=False,
        last_calibrate=False,
        bit_type=BitType,
        calibration_mode="layer_wise",
        observer_str="minmax",
        quantizer_str="uniform",
    ):
        super(HMQLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "linear_weight"
        self.observer = build_observer(
            self.module_type, self.bit_type, self.calibration_mode
        )
        self.quantizer = build_quantizer(self.bit_type, self.observer, self.module_type)

    def forward(self, x, x_quantizer=None, mp=None, layer_name='', quant_type='int8', node_type='mlp'):

        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params()
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight, False)
        bias = self.bias / (self.quantizer.scale * x_quantizer.scale)
        x_ln = F.linear(x, weight, bias)
        if mp is not None:
            mp.parse_layer(layer_name, node_type)
            mp.parse_ops('matmul', in_data=self.weight, in_data_name='', 
                        in_tmp=x, in_tmp_name='', 
                        out_data=x_ln, out_data_name='', 
                        feature_dict={'bias_shape': self.bias.shape, 'scale': 0.1, 'quant_type': quant_type}, int_ops=True)
        x = x_ln * x_quantizer.scale * self.quantizer.scale

        return x


class HMQAct(nn.Module):

    def __init__(
        self,
        quant=False,
        calibrate=False,
        last_calibrate=False,
        bit_type=BitType,
        calibration_mode="layer_wise",
        observer_str="minmax",
        quantizer_str="uniform",
    ):
        super(HMQAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "activation"
        self.observer = build_observer(
            self.module_type, self.bit_type, self.calibration_mode
        )
        self.quantizer = build_quantizer(self.bit_type, self.observer, self.module_type)

    def forward(self, x, dq=False):
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                self.quantizer.update_quantization_params()
        if not self.quant:
            return x
        x = self.quantizer(x, False)
        if dq:
            x = self.quantizer(x, True)
        return x


class HMQMulQK(nn.Module):

    def __init__(
        self,
        quant=False,
        calibrate=False,
        last_calibrate=False,
        bit_type=BitType,
        calibration_mode="layer_wise",
        observer_str="minmax",
        quantizer_str="uniform",
    ):
        super(HMQMulQK, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "activation"
        self.q_observer = build_observer(
            self.module_type, self.bit_type, self.calibration_mode
        )
        self.q_quantizer = build_quantizer(
            self.bit_type, self.q_observer, self.module_type
        )
        self.k_observer = build_observer(
            self.module_type, self.bit_type, self.calibration_mode
        )
        self.k_quantizer = build_quantizer(
            self.bit_type, self.k_observer, self.module_type
        )

    def forward(self, q, k, layer_name='', mp=None):
        if self.calibrate:
            self.q_quantizer.observer.update(q)
            self.k_quantizer.observer.update(k)
            if self.last_calibrate:
                self.q_quantizer.update_quantization_params(q)
                self.k_quantizer.update_quantization_params(k)
        if not self.quant:
            return q @ k.transpose(-2, -1)
        q = self.q_quantizer(q, False)
        k = self.k_quantizer(k, False)
        k_trans = k.transpose(-2, -1)
        x = q @ k_trans
        if mp is not None:
            mp.parse_layer(layer_name, 'kq_mul')
            mp.parse_ops('matmul', in_data=k_trans, in_data_name='', in_tmp=q, in_tmp_name='', out_data=x, out_data_name='', feature_dict={'scale': 0.1, 'quant_type': 'fp16'}, int_ops=True, keep_out=True)
        x = x * self.q_quantizer.scale * self.k_quantizer.scale

        return x


class HMQMulSV(nn.Module):

    def __init__(
        self,
        quant=False,
        calibrate=False,
        last_calibrate=False,
        bit_type=BitType,
        calibration_mode="layer_wise",
        observer_str="minmax",
        quantizer_str="uniform",
    ):
        super(HMQMulSV, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "activation"
        self.s_observer = build_observer(
            self.module_type, self.bit_type, self.calibration_mode
        )
        self.s_quantizer = build_quantizer(
            self.bit_type, self.s_observer, self.module_type
        )
        self.v_observer = build_observer(
            self.module_type, self.bit_type, self.calibration_mode
        )
        self.v_quantizer = build_quantizer(
            self.bit_type, self.v_observer, self.module_type
        )

    def forward(self, s, v, layer_name='', mp=None):
        if self.calibrate:
            self.s_quantizer.observer.update(s)
            self.v_quantizer.observer.update(v)
            if self.last_calibrate:
                self.s_quantizer.update_quantization_params(s)
                self.v_quantizer.update_quantization_params(v)
        if not self.quant:
            return s @ v
        s = self.s_quantizer(s, False)
        v = self.v_quantizer(v, False)
        x = s @ v
        x = x * self.s_quantizer.scale * self.v_quantizer.scale

        if mp is not None:
            mp.parse_layer(layer_name, 'vs_mul')
            mp.parse_ops('matmul', in_data=v, in_data_name='', in_tmp=s, in_tmp_name='', out_data=x
                    , out_data_name='', feature_dict={'scale': 0.1, 'quant_type': 'int8'}, int_ops=True)

        return x


class HMQGeLU(nn.GELU):
    def __init__(self):
        super(HMQGeLU, self).__init__()

    def forward(self, x, layer_name='', mp=None, C_dict=None):
        mp.parse_layer(layer_name, 'gelu')
        x = x.bfloat16()
        tmp_x = x
        pow_tmp = 0.044715 * torch.pow(x, 3)
        mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=x, in_tmp_name='A0', out_data=x, out_data_name='B0', verbose=True)
        mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=x, in_tmp_name='B0', out_data=pow_tmp, out_data_name='B0', verbose=True)
        mp.parse_ops('mul', in_data=torch.tensor([]), in_data_name='C5', in_tmp=pow_tmp, in_tmp_name='B0', out_data=pow_tmp, out_data_name='B0', verbose=True)
        C_dict['C4'] = torch.tensor(0.044715).bfloat16().item()

        pow_tmp = pow_tmp.bfloat16()
        pi_tmp = math.sqrt(2 / math.pi)
        pi_tmp = torch.tensor(pi_tmp).bfloat16()
        add_tmp = (x + pow_tmp).bfloat16()
        mp.parse_ops('add', in_data=x, in_data_name='A0', in_tmp=pow_tmp, in_tmp_name='B0', out_data=add_tmp, out_data_name='A1', verbose=True)

        x = pi_tmp * add_tmp
        mp.parse_ops('mul', in_data=add_tmp, in_data_name='A1', in_tmp=torch.tensor([]), in_tmp_name='C5', out_data=x, out_data_name='B0', verbose=True)
        C_dict['C5'] = torch.tensor(pi_tmp.item()).bfloat16().item()
        
        x = pade_tanh_tensor(x, mp, C_dict)  # Padé Approximation
        x = x.bfloat16()
        x += 1
        mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C6', in_tmp=x, in_tmp_name='B0', out_data=x, out_data_name='B0', verbose=True)
        mp.parse_ops('add', in_data=torch.tensor([]), in_data_name='C6', in_tmp=x, in_tmp_name='B0', out_data=x, out_data_name='B0', verbose=True)

        mul_tmp = tmp_x * x
        mp.parse_ops('mul', in_data=x, in_data_name='A0', in_tmp=tmp_x, in_tmp_name='B0', out_data=mul_tmp, out_data_name='A0', verbose=True)

        mul_tmp = mul_tmp.bfloat16()
        bfp_out = 0.5 * mul_tmp
        mp.parse_ops('mul', in_data=mul_tmp, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C6', out_data=bfp_out, out_data_name='A0', verbose=True, feature_dict={'quant_type': 'int8'})

        bfp_out = bfp_out.float()
        return bfp_out


class HMQSoftmax(nn.Module):
    def __init__(self):
        super(HMQSoftmax, self).__init__()

    def forward(self, x, layer_name='', mp=None, C_dict=None):
        if mp is not None:
            mp.parse_layer(layer_name, 'sftm')
            
        q = torch.floor(x * (1 / 0.6931471805599453))
        q = q.bfloat16()
        exp_x = 2**q
        exp_x = exp_x.bfloat16()

        exp_x_sum = torch.sum(exp_x, dim=-1, keepdim=True)
        exp_x_sum = exp_x_sum.bfloat16()
        if mp is not None:
            mp.parse_ops('mua', in_data=x, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='B0', out_data=exp_x_sum, out_data_name='A0', feature_dict={'constant': 'C1', 'constraint':['floor', 'rsh']})
            C_dict['C1'] = torch.tensor(1/0.6931471805599453).bfloat16().item()
        exp_xs = isqrt_sftm(exp_x_sum, mp)
        bfp_out = exp_x * exp_xs * exp_xs
        if mp is not None:
            mp.parse_ops('mmm', in_data=x, in_data_name='A0', out_data=bfp_out, out_data_name='A0', feature_dict={'y0_ref': 'C1', 'y1_ref': 'B0', 'y2_ref': 'B0', 'constraint':['floor', 'rsh'], 'quant_type': 'int8'})

        return bfp_out.float()


class HMQTanh(nn.Module):
    def __init__(self):
        super(HMQTanh, self).__init__()

    def forward(self, x):
        x = pade_tanh_tensor(x)  # Padé Approximation
        bfp_out = x.bfloat16()
        return bfp_out


class HMQLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(HMQLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x, layer_name='', mp=None, C_dict=None):

        if mp is not None:
            mp.parse_layer(layer_name, 'ln')
            C_dict['C2'] = torch.tensor(1/x.shape[-1]).bfloat16().item()
            C_dict['C3'] = torch.tensor(self.eps).bfloat16().item()

        assert self.normalized_shape == x.shape[-len(self.normalized_shape) :]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        x = x.bfloat16()
        x_sum = torch.sum(x, dim=dims, keepdim=True).bfloat16()
        mean = (x_sum / x.shape[-1]).bfloat16()
        

        pow_tmp = (x**2).bfloat16()
        pow_tmp_sum = torch.sum(pow_tmp, dim=dims, keepdim=True).bfloat16()
        mean_x2 = (pow_tmp_sum / pow_tmp.shape[-1]).bfloat16()
        if mp is not None:
            mp.parse_ops('ama', in_data=x, in_data_name='A0', in_tmp_name='B0', out_data=x_sum, out_data_name='A0', out_tmp_name='A1', verbose=True)
            mp.parse_ops('mul', in_data=x_sum, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C0', out_data=mean, out_data_name='B0', verbose=True)
            mp.parse_ops('mul', in_data=pow_tmp_sum, in_data_name='A1', in_tmp=torch.tensor([]), in_tmp_name='C0', out_data=mean_x2, out_data_name='A0', verbose=True)

        mean_pow_tmp = (mean**2).bfloat16()
        var = (mean_x2 - mean_pow_tmp).bfloat16()
        add_tmp = (var + torch.tensor(self.eps).bfloat16()).float()
        if mp is not None:
            mp.parse_ops('mul', in_data=mean, in_data_name='B0', in_tmp=mean, in_tmp_name='B0', out_data=mean_pow_tmp, out_data_name='B1', verbose=True)
            mp.parse_ops('add', in_data=mean_x2, in_data_name='A0', in_tmp=mean_pow_tmp, in_tmp_name='B1', out_data=var, out_data_name='A0', feature_dict={"sub": "Y1"}, verbose=True)
            mp.parse_ops('add', in_data=var, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C1', out_data=add_tmp, out_data_name='A0', verbose=True)
        

        var_sqrt = isqrt_ln(add_tmp, mp, C_dict).to("cuda").bfloat16()  # Inverse Sqrt
        sub_tmp = (x - mean).bfloat16()
        x_norm = (sub_tmp * var_sqrt).bfloat16()

        if self.elementwise_affine:
            w_x_tmp = (self.weight * x_norm).bfloat16()
            x_norm = w_x_tmp + self.bias.bfloat16()

        mean_shape = mean.transpose(1, 2).shape
        var_sqrt_shape = var_sqrt.transpose(1, 2).shape

        if mp is not None:
            mp.parse_ops('amd', in_data=x, in_data_name='A0', out_data=x_norm, out_data_name='A0', verbose=True, feature_dict={'y0': mean_shape[1:], 'y1': var_sqrt_shape[1:], 'y2': self.weight.shape, 'y3': self.bias.shape, 'yrf0': 'B0', 'yrf1': 'B1', 'yrf2': 'BW', 'yrf3': 'BB', 'quant_type': 'int8'})

        bfp_out = x_norm.float()

        return bfp_out
