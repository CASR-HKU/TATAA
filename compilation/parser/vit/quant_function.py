import torch
from torch.autograd import Function
import torch.nn.functional as F
from hlbfp_format import HybridLowBlockFP
from para_config import *

def calc_padding(fold, dim):
    if fold>=dim:
        nstack = 1
    else:
        quo,rem = divmod(int(dim), int(fold))
        nstack = quo+(rem>0)
    num = nstack*fold
    p = num-dim
    return int(p)

class BFPQuantFunction(Function):
    """Quantization function for block floating point, supporting fp32, lfp (low-precision fp) and bfp

    For Vision Transformers, Shapes of
    Activations:
        - Conv2D (embedding): [B, C, H, W]
        - Linear (attention): [B, N, C]
        - Linear (other linears): [B, C]
    Weights:
        - Conv2D (embedding): [C_out, C_in, H, W]
        - Linear (attention): [C_out, C_in]
        - Linear (other linears): [C_out, C_in]
    Bias:
        - Conv2D (embedding): [C_out]
        - Linear (attention): [C_out]
        - Linear (other linears): [C_out]

    """

    @staticmethod
    def forward(ctx, x, bfp_format: HybridLowBlockFP, pt_flag=False):
        ctx.fp_type = (
            "fp32" if (bfp_format.exp_bits == -1) or (bfp_format.man_bits == -1) else ("bfp")
        )
        if ctx.fp_type == "fp32":
            return x
        elif ctx.fp_type == "bfp":
            bfp_values, shared_exp = convert_tensor_to_bfp(x, bfp_format)
            return bfp_values

def convert_tensor_to_bfp(x: torch.Tensor, bfp: HybridLowBlockFP):
    assert 4 >= len(x.shape) >= 1, "Shape not supported"
    if bfp.exp_bits == -1 or bfp.man_bits == -1:
        raise RuntimeError("No need to call this function for fp32!")

    x_abs = torch.abs(x)
    x_sign = torch.sign(x)

    # Convert to BFP based on the block size
    # Case 1: x: [B, C, H, W]
    if len(x.shape) == 4:
        bfp_values, shared_exp = bfp_TD4(x_abs, bfp)
        bfp_values = bfp_values * x_sign
        return bfp_values, shared_exp
    # Case 2: x: [B, N, C]
    elif len(x.shape) == 3:
        bfp_values, shared_exp = bfp_TD3(x_abs, bfp)
        bfp_values = bfp_values * x_sign
        return bfp_values, shared_exp
    # Case 3: x: [B, C]
    elif len(x.shape) == 2:
        bfp_values, shared_exp = bfp_TD2(x_abs, bfp)
        bfp_values = bfp_values * x_sign
        return bfp_values, shared_exp
    # Case 4: x: [Cout] only used for bias
    elif len(x.shape) == 1:
        bfp_values, shared_exp = bfp_TD1(x_abs, bfp)
        bfp_values = bfp_values * x_sign
        return bfp_values, shared_exp


def bfp_TD4(x: torch.Tensor, bfp: HybridLowBlockFP):
    """Convert a 4 dimension tensor to BFP format (TD: Tensor Dimension)

    Args:
        x (torch.Tensor): input tensor (must be abs)
        bfp (BlockFloatingPoint): block floating point format

    Returns:
        x_bfp: block floating point tensor (abs)

    NOTE: for the TD4 case, the block size can only be 1, 0 or -1.
    """
    original_shape = x.shape
    dim = original_shape
    assert len(original_shape) == 4, "Shape is not 4!"
    # channel-wise, i.e. merge HW dimension

    if bfp.block_size == 0:
        x = x.view(original_shape[0], original_shape[1], -1)
        mode_dim = 2
    # batch-wise, i.e. merge CHW dimension
    elif bfp.block_size == 1:
        x = x.view(original_shape[0], -1)
        mode_dim = 1
    # whole tensor, i.e. merge BCHW dimension
    elif bfp.block_size == -1:
        x = x.view(-1)
        mode_dim = 0
    else:
        f0,f1 = BLOCK_DIM,BLOCK_DIM
        p0,p1,p2,p3 = -1,-1,f1,f0
        factors = [p0,p1,p2,p3]
        dim_len = len(original_shape)
        fact = [factors[i] if factors[i] != -1 else dim[i] for i in range(dim_len)]
        num_pad = [calc_padding(fact[i], dim[i]) for i in range(dim_len)]
        padding =tuple([0,num_pad[3],0,num_pad[2]])
        data_pad = F.pad(input=x, pad=padding, mode='constant', value=0)
        pad_shape = data_pad.shape
        data_unf = data_pad.unfold(2, f0, f1).unfold(3, f0, f1)
        x = data_unf.contiguous().view([data_unf.size(0), data_unf.size(1), data_unf.size(2)*data_unf.size(3), -1])
        mode_dim = 3

    if bfp.block_size == 0: 
        x_abs_man, x_abs_exp = torch.frexp(x)
        shared_exp, _ = torch.max(x_abs_exp, dim=mode_dim, keepdim=True)

        interval = torch.exp2(shared_exp - bfp.man_bits)
        bfp_min, bfp_max = bfp.get_hlbfp_value_range(shared_exp)
        x = torch.clamp(x, min=bfp_min, max=bfp_max)
        x_bfp = torch.round(x / interval) * interval

        x_bfp = x_bfp.view(original_shape)

        assert x_bfp.shape == original_shape, "Shape mismatch!"
    else:
        x_abs_man, x_abs_exp = torch.frexp(x)
        shared_exp, _ = torch.max(x_abs_exp, dim=mode_dim, keepdim=True)

        interval = torch.exp2(shared_exp - bfp.man_bits)
        bfp_min, bfp_max = bfp.get_hlbfp_value_range(shared_exp)
        x = torch.clamp(x, min=bfp_min, max=bfp_max)
        x_bfp = torch.round(x / interval) * interval

        test = x_bfp.unfold(3,f0,f1).unfold(2,int(pad_shape[3]/f0),int(pad_shape[3]/f1))
        test_trans = test.transpose(-2,-1)
        test_trans_shape = test_trans.shape
        test_reshape = test_trans.reshape(test_trans_shape[0], test_trans_shape[1], test_trans_shape[2]*test_trans_shape[3], -1)
        data_out = test_reshape[:, :, :dim[2], :dim[3]].contiguous()
        x_bfp = data_out

    return x_bfp, shared_exp


def bfp_TD3(x: torch.Tensor, bfp: HybridLowBlockFP):
    """Convert a 3 dimension tensor to BFP format (TD: Tensor Dimension)
    NOTE: for the TD3 case, the block size can only be 1, 0, -1 or -2.
    """
    # Step 1: Merge the dimensions (NOTE: in the future we may need to make blocks in C dimension)
    original_shape = x.shape
    dim = original_shape
    # print("TD3 X Shape: ", original_shape)
    assert len(original_shape) == 3, "Shape is not 3!"
    # vector-wise
    if bfp.block_size == 0:
        mode_dim = 2
    # batch-wise, i.e. merge NC dimension
    elif bfp.block_size == 1:
        mode_dim = 1
        x = x.view(original_shape, -1)
    # whole tensor, i.e. merge BNC dimension
    elif bfp.block_size == -1:
        mode_dim = 0
        x = x.view(-1)
    else:
        f0,f1 = BLOCK_DIM,BLOCK_DIM
        p0,p1,p2 = -1,f1,f0
        factors = [p0,p1,p2]
        dim_len = len(original_shape)
        fact = [factors[i] if factors[i] != -1 else dim[i] for i in range(dim_len)]
        num_pad = [calc_padding(fact[i], dim[i]) for i in range(dim_len)]
        padding =tuple([0,num_pad[2],0,num_pad[1]])
        data_pad = F.pad(input=x, pad=padding, mode='constant', value=0)
        pad_shape = data_pad.shape
        data_unf = data_pad.unfold(1, f0, f1).unfold(2, f0, f1)
        x = data_unf.contiguous().view([data_unf.size(0), data_unf.size(1)*data_unf.size(2), -1])
        mode_dim = 2

    if bfp.block_size == 0:
        x_abs_man, x_abs_exp = torch.frexp(x)
        shared_exp, _ = torch.max(x_abs_exp, dim=mode_dim, keepdim=True)

        interval = torch.exp2(shared_exp - bfp.man_bits)
        bfp_min, bfp_max = bfp.get_hlbfp_value_range(shared_exp)
        x = torch.clamp(x, min=bfp_min, max=bfp_max)
        x_bfp = torch.round(x / interval) * interval

        x_bfp = x_bfp.view(original_shape)
        assert x_bfp.shape == original_shape, "Shape mismatch!"
    else:
        x_abs_man, x_abs_exp = torch.frexp(x)
        shared_exp, _ = torch.max(x_abs_exp, dim=mode_dim, keepdim=True)

        interval = torch.exp2(shared_exp - bfp.man_bits)
        bfp_min, bfp_max = bfp.get_hlbfp_value_range(shared_exp)
        x = torch.clamp(x, min=bfp_min, max=bfp_max)
        x_bfp = torch.round(x / interval) * interval

        test = x_bfp.unfold(2,f0,f1).unfold(1,int(pad_shape[2]/f0),int(pad_shape[2]/f1))
        test_trans = test.transpose(-2,-1)
        test_trans_shape = test_trans.shape
        test_reshape = test_trans.reshape(test_trans_shape[0], test_trans_shape[1]*test_trans_shape[2], -1)
        data_out = test_reshape[:, :dim[1], :dim[2]].contiguous()
        x_bfp = data_out

    return x_bfp, shared_exp


def bfp_TD2(x: torch.Tensor, bfp: HybridLowBlockFP):
    """Convert a 2 dimension tensor to BFP format (TD: Tensor Dimension)
    NOTE: for the TD2 case, the block size can be any value.
    """
    # Step 1: Merge the dimensions (NOTE: in the future we may need to make blocks in C dimension)
    original_shape = x.shape
    dim = original_shape
    assert len(original_shape) == 2, "Shape is not 2!"
    # batch-wise
    if bfp.block_size == 0:
        mode_dim = 1
    # whole tensor, i.e. merge BNC dimension
    elif bfp.block_size == -1:
        x = x.view(-1)
        mode_dim = 0
    else:
        f0,f1 = BLOCK_DIM,BLOCK_DIM
        p0,p1 = f1,f0
        factors = [p0,p1]
        dim_len = len(original_shape)
        fact = [factors[i] if factors[i] != -1 else dim[i] for i in range(dim_len)]
        num_pad = [calc_padding(fact[i], dim[i]) for i in range(dim_len)]
        padding =tuple([0,num_pad[1],0,num_pad[0]])
        data_pad = F.pad(input=x, pad=padding, mode='constant', value=0)
        pad_shape = data_pad.shape
        data_unf = data_pad.unfold(0, f0, f1).unfold(1, f0, f1)
        x = data_unf.contiguous().view([data_unf.size(0)*data_unf.size(1), -1])
        mode_dim = 1

    if bfp.block_size == 0:
        x_abs_man, x_abs_exp = torch.frexp(x)
        shared_exp, _ = torch.max(x_abs_exp, dim=mode_dim, keepdim=True)
        interval = torch.exp2(shared_exp - bfp.man_bits)
        bfp_min, bfp_max = bfp.get_hlbfp_value_range(shared_exp)
        x = torch.clamp(x, min=bfp_min, max=bfp_max)
        x_bfp = torch.round(x / interval) * interval

        x_bfp = x_bfp.view(original_shape)
        assert x_bfp.shape == original_shape, "Shape mismatch!"
    else:
        x_abs_man, x_abs_exp = torch.frexp(x)
        shared_exp, _ = torch.max(x_abs_exp, dim=mode_dim, keepdim=True)

        interval = torch.exp2(shared_exp - bfp.man_bits)
        bfp_min, bfp_max = bfp.get_hlbfp_value_range(shared_exp)
        x = torch.clamp(x, min=bfp_min, max=bfp_max)
        x_bfp = torch.round(x / interval) * interval
        test = x_bfp.unfold(1,f0,f1).unfold(0,int(pad_shape[1]/f0),int(pad_shape[1]/f1))
        test_trans = test.transpose(-2,-1)
        test_trans_shape = test_trans.shape
        test_reshape = test_trans.reshape(test_trans_shape[0]*test_trans_shape[1], -1)
        data_out = test_reshape[:dim[0], :dim[1]].contiguous()
        x_bfp = data_out

    return x_bfp, shared_exp

def bfp_TD1(x: torch.Tensor, bfp: HybridLowBlockFP):
    """Convert a 1 dimension tensor to BFP format (TD: Tensor Dimension), only used for bias"""
    x_abs_man, x_abs_exp = torch.frexp(x)
    shared_exp, _ = torch.max(x_abs_exp, dim=0, keepdim=True)
    interval = torch.exp2(shared_exp - bfp.man_bits)
    bfp_min, bfp_max = bfp.get_hlbfp_value_range(shared_exp)
    x = torch.clamp(x, min=bfp_min, max=bfp_max)
    x_bfp = torch.round(x / interval) * interval
    return x_bfp, shared_exp