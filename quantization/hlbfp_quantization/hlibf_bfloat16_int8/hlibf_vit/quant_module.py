import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import BitType
from build_observer import *
from build_quantizer import *

# def conventional_cordic_exp(z: torch.Tensor, iter_num=32):
#     y = torch.zeros(z.shape)
#     x = torch.zeros(z.shape)

#     y = y.to('cuda')
#     x = x.to('cuda')

#     if iter_num == 8:
#         x += 1.2109375
#     if iter_num == 16:
#         x += 1.207489013671875
#     if iter_num == 24:
#         x += 1.207497000694275
#     if iter_num == 32:
#         x += 1.2074970677495

#     x = x.bfloat16()

#     list_thetai = [0.549306144334055, 0.255412811882995, 0.125657214140453, 0.0625815714770030,
#                    0.0312601784906670, 0.0156262717520522, 0.00781265895154042, 0.00390626986839683,
#                    0.00195312748353255, 0.000976562810441036, 0.000488281288805113, 0.000244140629850639,
#                    0.000122070313106330, 6.10351563257912e-05, 3.05175781344739e-05, 1.52587890636842e-05,
#                    7.62939453139803e-06, 3.81469726564350e-06, 1.90734863281481e-06, 9.53674316406539e-07,
#                    4.76837158203161e-07, 2.38418579101567e-07, 1.19209289550782e-07, 5.96046447753907e-08,
#                    2.98023223876953e-08, 1.49011611938477e-08, 7.45058059692383e-09, 3.72529029846191e-09,
#                    1.86264514923096e-09, 9.31322574615479e-10, 4.65661287307739e-10, 2.32830643653870e-10]

#     for i in range(iter_num):
#         signz = z.sign()
#         y_pow = y * signz * math.pow(2, -(i + 1))
#         y_pow = y_pow.bfloat16()
#         x += y_pow
#         x = x.bfloat16()
#         x_tmp = x - y_pow
#         x_tmp = x_tmp.bfloat16()
#         y += (x_tmp) * signz * math.pow(2, -(i + 1))
#         y = y.bfloat16()
#         z -= signz * list_thetai[i]
#         z = z.bfloat16()

#     return x + y


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

def isqrt(x):
    y = x.bfloat16()
    i = y.view(torch.short)
    i = torch.tensor(0x5f37).short() - (i >> 1).short()
    y = i.view(torch.bfloat16)
    
    y_tmp = (y * y).bfloat16()
    x = (x * 0.5).bfloat16()

    x_tmp = x.bfloat16()
    mul_tmp = x_tmp * y_tmp
    mul_tmp = mul_tmp.bfloat16()
    sub_tmp = 1.5 - mul_tmp
    sub_tmp = sub_tmp.bfloat16()
    y = y * sub_tmp
    return y

class HMQConv2d(nn.Conv2d):

    def __init__(self,
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
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
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

        self.module_type = 'conv_weight'
        self.observer = build_observer(self.module_type, self.bit_type, self.calibration_mode)
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
        x = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        x = x * x_quantizer.scale * self.quantizer.scale

        return x

class HMQLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BitType,
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(HMQLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        self.observer = build_observer(self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.bit_type, self.observer, self.module_type)

    def forward(self, x, x_quantizer=None):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight, False)
        bias = self.bias / (self.quantizer.scale * x_quantizer.scale)
        x = F.linear(x, weight, bias)
        x = x * x_quantizer.scale * self.quantizer.scale

        return x

class HMQAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BitType,
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(HMQAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.bit_type, self.observer, self.module_type)

    def forward(self, x, dq=True):
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return x
        x = self.quantizer(x, False)
        if dq:
            x = self.quantizer(x, True)
        return x

class HMQMulQK(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BitType,
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(HMQMulQK, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.q_observer = build_observer(self.module_type, self.bit_type, self.calibration_mode)
        self.q_quantizer = build_quantizer(self.bit_type, self.q_observer, self.module_type)
        self.k_observer = build_observer(self.module_type, self.bit_type, self.calibration_mode)
        self.k_quantizer = build_quantizer(self.bit_type, self.k_observer, self.module_type)

    def forward(self, q, k):
        if self.calibrate:
            self.q_quantizer.observer.update(q)
            self.k_quantizer.observer.update(k)
            if self.last_calibrate:
                self.q_quantizer.update_quantization_params(q)
                self.k_quantizer.update_quantization_params(k)
        if not self.quant:
            return q @ k
        q = self.q_quantizer(q, False)
        k = self.k_quantizer(k, False)
        x = q @ k
        x = x * self.q_quantizer.scale * self.k_quantizer.scale

        return x

class HMQMulSV(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BitType,
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(HMQMulSV, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.s_observer = build_observer(self.module_type, self.bit_type, self.calibration_mode)
        self.s_quantizer = build_quantizer(self.bit_type, self.s_observer, self.module_type)
        self.v_observer = build_observer(self.module_type, self.bit_type, self.calibration_mode)
        self.v_quantizer = build_quantizer(self.bit_type, self.v_observer, self.module_type)

    def forward(self, s, v):
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

        return x

class HMQGeLU(nn.GELU):
    def __init__(self):
        super(HMQGeLU, self).__init__()

    def forward(self, x):
        x = x.bfloat16()
        tmp_x = x
        pow_tmp = 0.044715*torch.pow(x, 3)
        pow_tmp = pow_tmp.bfloat16()
        pi_tmp = math.sqrt(2/math.pi)
        pi_tmp = torch.tensor(pi_tmp).bfloat16()
        add_tmp = (x + pow_tmp).bfloat16()
        x = pi_tmp*add_tmp
        x = pade_tanh_tensor(x) # Padé Approximation
        x = x.bfloat16()
        x += 1
        mul_tmp = tmp_x * x
        mul_tmp = mul_tmp.bfloat16()
        bfp_out = 0.5 * mul_tmp
        bfp_out = bfp_out.float()
        return bfp_out

class HMQSoftmax(nn.Module):
    def __init__(self):
        super(HMQSoftmax, self).__init__()

    def forward(self, x):

        q = torch.floor( x * (1 / 0.6931471805599453) )
        q = q.bfloat16()
        exp_x = (2 ** q).bfloat16()

        # q = torch.floor( x / 0.6931471805599453 ).bfloat16()
        # q_tmp = (q * 0.6931471805599453).bfloat16()
        # r = (x - q_tmp).bfloat16()
        # cordic_exp_x = conventional_cordic_exp(r)
        # exp_x = (2 ** q * cordic_exp_x).bfloat16()

        exp_x_sum = torch.sum(exp_x, dim=-1, keepdim=True)
        exp_x_sum = exp_x_sum.bfloat16()

        exp_xs = isqrt(exp_x_sum)
        bfp_out = exp_x * exp_xs * exp_xs
        
        bfp_out = bfp_out.float()

        return bfp_out

class HMQLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(HMQLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        x = x.bfloat16()
        x_sum = torch.sum(x, dim=dims, keepdim=True).bfloat16()
        mean = (x_sum / x.shape[-1]).bfloat16()
 
        pow_tmp = (x ** 2).bfloat16()
        pow_tmp_sum = torch.sum(pow_tmp, dim=dims, keepdim=True).bfloat16()
        mean_x2 = (pow_tmp_sum / pow_tmp.shape[-1]).bfloat16()
 
        mean_pow_tmp = (mean ** 2).bfloat16()
        var = (mean_x2 - mean_pow_tmp).bfloat16()
        add_tmp = (var + torch.tensor(self.eps).bfloat16()).float()
        var_sqrt = isqrt(add_tmp).to('cuda').bfloat16() # Inverse Sqrt
        sub_tmp = (x - mean).bfloat16()
        x_norm = (sub_tmp * var_sqrt).bfloat16()

        if self.elementwise_affine:
            w_x_tmp = (self.weight * x_norm).bfloat16()
            x_norm = w_x_tmp + self.bias.bfloat16()

        bfp_out = x_norm.float()

        return bfp_out