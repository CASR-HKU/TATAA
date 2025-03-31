import numpy as np
import torch
from itertools import product

class HybridLowBlockFP:
    """Block Floating Point Format. It supports regular low precision FP without block (i.e., block_size=-2) as well.

    Args:
        block_size (int): block size.
            * For activations: 1 for batch-wise, 0 for channel-wise (in the embedding) or vector-wise (in the attention & linear),
            -1 for the whole tensor, -2 for element-wise (No block). Other numbers: block size.
            * For 4D weights: 1 for kernel-wise, 0 for channel-wise, -1 for the whole tensor, -2 for element-wise (No block). Other numbers: block size.
            * For 2D weights: 0 for input-wise, -1 for the whole tensor, -2 for the element-wise (No block). Other numbers: block size.
            * For bias: -1 for the whole tensor, -2 for element-wise (No block). Other numbers: block size.
        exp_bits (int): number of bits for the shared exponent. -1: original model with fp32
        man_bits (int): number of bits for the mantissa. -1: original model with fp32
        underflow (str): underflow mode. "zero" or "min". Zero means the smallest number is 0,
            min means the smallest number is the smallest number in the format
        round_mode (str): round mode. "nearest" or "stochastic"

    NOTE:
        1. For now, we only support subnormalized mantissa (i.e., 0.mantissa), because in block floating point,
            it is hard to distinguish between subnormalized and normalized numbers
        2. In the next version, add support for self-defined block size with padding
        3. Block sized except [1, 0, -1, -2] can only be used for 2D weights
    """

    def __init__(
        self,
        block_size=0,
        exp_bits=4,  # For delta_encoding, the exp_bits refers to de, for lfp, the exp_bits refers to the common exp
        man_bits=7,
        exp_bias=7,  # only used for -2 block_size (No block, lfp)
        data_type="bfp",
        underflow="min",
        round_mode="nearest",
    ):
        assert round_mode in [
            "nearest",
            "stochastic",
        ], "round_mode must be nearest or stochastic"

        self.block_size = block_size
        self.exp_bits = exp_bits
        self.man_bits = man_bits
        self.exp_bias = exp_bias
        self.data_type = data_type
        self.underflow = underflow
        self.round_mode = round_mode
        self.bfp_init = True

    def __str__(self):
        if self.exp_bits == -1 or self.man_bits == -1:
            return "Original Model (pytorch fp32)"
        elif self.data_type == "bfloat":
            return "Brain Floating Point (exponent={:d}, mantissa={:d}, exp_bias={:d}, block={:d})".format(
                self.exp_bits, self.man_bits, self.exp_bias, self.block_size
            )
        else:
            return "Block Floating Point 16 (exponent={:d}, mantissa={:d}, block={:d})".format(
                self.exp_bits, self.man_bits, self.block_size
            )

    def __repr__(self):
        if self.exp_bits == -1 or self.man_bits == -1:
            return "Original Model (pytorch fp32)"
        elif self.data_type == "bfloat":
            return "Brain Floating Point 16 (exponent={:d}, mantissa={:d}, exp_bias={:d}, block={:d})".format(
                self.exp_bits, self.man_bits, self.exp_bias, self.block_size
            )
        else:
            return "Block Floating Point (exponent={:d}, mantissa={:d}, block={:d})".format(
                self.exp_bits, self.man_bits, self.block_size
            )

    def get_hlbfp_value_range(self, shared_exp):
        """Get the value range of the bfp format given the shared exponent"""

        # Max and Min are both positive. For the negative numbers, they are symmetric
        max_value = (2 ** (-self.man_bits) * (2 ** (self.man_bits) - 1)) * torch.exp2(
            shared_exp
        )
        min_value = 2 ** (-self.man_bits) * torch.exp2(shared_exp)
        return min_value, max_value

    def generate_all_bfp_values(self, shared_exp):
        """Generate all possible values in the bfp format given the shared exponent

        NOTE: can only be used for one dimension (you need iteration for multiple dimensions)

        Args:
            shared_exp: Shared exponent in one dimension

        Returns:
            res: all possible values
        """
        all_values = []
        for S in [-1.0, 1.0]:
            for F_str_iter in product(*[[0, 1]] * self.man_bits):
                F_str = "".join(str(i) for i in F_str_iter)

                F_enc = (
                    sum([2 ** (-(i + 1)) * int(a) for i, a in enumerate(F_str)])
                    * 2 ** (len(F_str))
                    * 2 ** (-self.man_bits)
                )
                # for now, all numbers are subnormalized, i.e., 0.mantissa
                F_eff = F_enc

                fp8_val = S * 2 ** (shared_exp) * F_eff
                all_values.append(fp8_val)
        res = np.array(all_values)
        res = np.sort(res)
        return res
