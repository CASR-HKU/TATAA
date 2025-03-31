import numpy as np
from hlbfp_format import HybridLowBlockFP

""" Naming definition (from bottom up):
    - HybridLowBlockFP: the class for hlbfp format (activation, weight and bias). The lowest level.
    - layer_format: the format for each layer, which is a list of HybridLowBlockFP objects. e.g., [act, w, bias] or [act]
    - format_dict: the layer_format for each block in ViT. e.g., attn_format_dict={"qkv": layer_format0, "mulqk": layer_format1, ...}
    - model_format: the format for all layers, which is a list of format_dict. 
    
    Finally, the model_format should be like [embed_format_dist, attn_format_dicts, mlp_format_dicts, head_format_dict]

    NOTE: 
    - The model_format will be finally applied to the model
    - attn_format_dicts & mlp_format_dicts are lists of format_dict, which are the format for each block
    - Lists of attn_format_dict & mlp_format_dict should have the same length as blk_depth
"""

""" FormatConfig definition:
    layer_config = [[act_block_size, act_exp_bits, act_man_bits, act_exp_bias],
                    [w_block_size, w_exp_bits, w_man_bits, w_exp_bias],
                    [bias_block_size, bias_exp_bits, bias_man_bits, bias_exp_bias]], length = 3
    or:
    layer_config = [act_block_size, act_exp_bits, act_man_bits, act_exp_bias], length = 4

    The layer_config_to_layer_format function is used to convert layer_config to HybridLowBlockFP objects (layer_format level).

    model_config_dict is a one-dim dict that contains all the layer_format for all layers.
    model_config_list is a one-dim list that contains all the layer_config for all layers.

    NOTE: 
    - model_format is used for the final model, model_config_dict and model_config_list are used for debugging, mixed-precision search, etc.
    - model_config_dict should be converted to model_format before applying to the model
"""

def layer_config_to_layer_format(layer_config: list):
    if len(layer_config) == 4:
        # activation only
        [act_block_size, act_exp_bits, act_man_bits, act_exp_bias] = layer_config
        hlbfp_format_act = HybridLowBlockFP(
            block_size=act_block_size,
            exp_bits=act_exp_bits,
            man_bits=act_man_bits,
            exp_bias=act_exp_bias,
            underflow="min",
            round_mode="nearest",
        )
        hlbfp_format_list = [hlbfp_format_act]
        return hlbfp_format_list
    elif len(layer_config) == 3:
        # act, w, bias, for linear and conv2d
        act_config = layer_config[0]
        w_config = layer_config[1]
        bias_config = layer_config[2]
        [act_block_size, act_exp_bits, act_man_bits, act_exp_bias] = act_config
        [w_block_size, w_exp_bits, w_man_bits, w_exp_bias] = w_config
        [bias_block_size, bias_exp_bits, bias_man_bits, bias_exp_bias] = bias_config
        hlbfp_format_act = HybridLowBlockFP(
            block_size=act_block_size,
            exp_bits=act_exp_bits,
            man_bits=act_man_bits,
            exp_bias=act_exp_bias,
            underflow="min",
            round_mode="nearest",
        )
        hlbfp_format_weight = HybridLowBlockFP(
            block_size=w_block_size,
            exp_bits=w_exp_bits,
            man_bits=w_man_bits,
            exp_bias=w_exp_bias,
            underflow="min",
            round_mode="nearest",
        )
        hlbfp_format_bias = HybridLowBlockFP(
            block_size=bias_block_size,
            exp_bits=bias_exp_bits,
            man_bits=bias_man_bits,
            exp_bias=bias_exp_bias,
            underflow="min",
            round_mode="nearest",
        )
        hlbfp_layer_format = [hlbfp_format_act, hlbfp_format_weight, hlbfp_format_bias]
        return hlbfp_layer_format
    else:
        raise NotImplementedError("Config array is not supported yet")

class FormatConfig:
    """Under construction"""

    def __init__(
        self, model_category="deit", blk_depth=12, shared_exp_bits=4, bfp_block_size=0
    ):
        """The format config for all layers

        Args:
            model_category (list): model name
            shared_exp_bits (int, optional): The shared exp bits for all layers. Defaults to 4.

        NOTE:
            - Each element in layer_str_arrays is a string, which is the layer name.
            - For embedding (conv2d) and linear layers (named embed), act, w and bias are all considered
            - For act_out and residual layers, only act is considered
            - For now, we only target vision transformers
        """
        # TODO: add more models including LLM
        # for ViT, the list is embed -> block1 -> block2 -> ... -> block12 -> head
        self.model_category = model_category
        self.blk_depth = blk_depth
        self.shared_exp_bits = shared_exp_bits
        self.bfp_block_size = bfp_block_size

    def init_format(
        self, init_bfp_bit=7, init_lfp_exp=4, init_lfp_man=3, init_lfp_bias=7
    ):
        """Initialize the format for all layers, default all mantissa bits to 7
        The list is for mixed-precision search

        NOTE: by default, all the activations are lfp, weights are bfp and bias are fp32
        """
        model_config_dict = {}
        model_config_list = []
        if self.model_category == "deit":
            # act, w, bias
            model_config_dict["embed"] = [
                [-2, init_lfp_exp, init_lfp_man, init_lfp_bias],
                [self.bfp_block_size, self.shared_exp_bits, init_bfp_bit, 0],
                # [0, -1, -1, init_lfp_bias],
                [8, self.shared_exp_bits, init_bfp_bit, init_lfp_bias],
            ]
            model_config_list.append(model_config_dict["embed"])
            for i in range(1, self.blk_depth + 1):
                # act, w, bias
                model_config_dict["blk" + str(i) + "qkv"] = [
                    [-2, init_lfp_exp, init_lfp_man, init_lfp_bias],
                    [self.bfp_block_size, self.shared_exp_bits, init_bfp_bit, 0],
                    # [0, -1, -1, init_lfp_bias],
                    [8, self.shared_exp_bits, init_bfp_bit, init_lfp_bias],
                ]
                # act only
                model_config_dict["blk" + str(i) + "mulqk"] = [
                    -2,
                    init_lfp_exp,
                    init_lfp_man,
                    init_lfp_bias,
                ]
                # act only
                model_config_dict["blk" + str(i) + "sftm"] = [
                    -2,
                    init_lfp_exp,
                    init_lfp_man,
                    init_lfp_bias,
                ]
                # act only
                model_config_dict["blk" + str(i) + "mulsv"] = [
                    -2,
                    init_lfp_exp,
                    init_lfp_man,
                    init_lfp_bias,
                ]
                # act, w, bias
                model_config_dict["blk" + str(i) + "proj"] = [
                    [-2, init_lfp_exp, init_lfp_man, init_lfp_bias],
                    [self.bfp_block_size, self.shared_exp_bits, init_bfp_bit, 0],
                    # [0, -1, -1, init_lfp_bias],
                    [8, self.shared_exp_bits, init_bfp_bit, init_lfp_bias],
                ]
                # act, w, bias
                model_config_dict["blk" + str(i) + "fc1"] = [
                    [-2, init_lfp_exp, init_lfp_man, init_lfp_bias],
                    [self.bfp_block_size, self.shared_exp_bits, init_bfp_bit, 0],
                    # [0, -1, -1, init_lfp_bias],
                    [8, self.shared_exp_bits, init_bfp_bit, init_lfp_bias],
                ]
                # act, w, bias
                model_config_dict["blk" + str(i) + "fc2"] = [
                    [-2, init_lfp_exp, init_lfp_man, init_lfp_bias],
                    [self.bfp_block_size, self.shared_exp_bits, init_bfp_bit, 0],
                    # [0, -1, -1, init_lfp_bias],
                    [8, self.shared_exp_bits, init_bfp_bit, init_lfp_bias],
                ]
                model_config_list.append(model_config_dict["blk" + str(i) + "qkv"])
                model_config_list.append(model_config_dict["blk" + str(i) + "mulqk"])
                model_config_list.append(model_config_dict["blk" + str(i) + "sftm"])
                model_config_list.append(model_config_dict["blk" + str(i) + "mulsv"])
                model_config_list.append(model_config_dict["blk" + str(i) + "proj"])
                model_config_list.append(model_config_dict["blk" + str(i) + "fc1"])
                model_config_list.append(model_config_dict["blk" + str(i) + "fc2"])
            # act, w, bias
            model_config_dict["head"] = [
                [-2, init_lfp_exp, init_lfp_man, init_lfp_bias],
                [self.bfp_block_size, self.shared_exp_bits, init_bfp_bit, 0],
                # [0, -1, -1, init_lfp_bias],
                [8, self.shared_exp_bits, init_bfp_bit, init_lfp_bias],
            ]
            model_config_list.append(model_config_dict["head"])
        else:
            raise NotImplementedError("Other models are not supported yet")
        return model_config_dict, model_config_list

    def config_list_to_dict(self, model_config_list: list):
        """Generate the format for all layers from the config array

        Args:
            model_config_list (list): see the definition in the beginning of this file

        NOTE: by default, all the activations are lfp, weights are bfp, and bias are fp32

        """
        model_config_dict = {}
        if self.model_category == "deit":
            config_array_len = (
                1 + self.blk_depth * 7 + 1
            )  # 1 for embed, 7 for each block, 1 for head
            assert (
                len(model_config_list) == config_array_len
            ), "The length of config array is not correct"
            model_config_dict["embed"] = model_config_list[0]
            for i in range(1, self.blk_depth + 1):
                model_config_dict["blk" + str(i) + "qkv"] = model_config_list[i * 7 - 6]
                model_config_dict["blk" + str(i) + "mulqk"] = model_config_list[
                    i * 7 - 5
                ]
                model_config_dict["blk" + str(i) + "sftm"] = model_config_list[
                    i * 7 - 4
                ]
                model_config_dict["blk" + str(i) + "mulsv"] = model_config_list[
                    i * 7 - 3
                ]
                model_config_dict["blk" + str(i) + "proj"] = model_config_list[
                    i * 7 - 2
                ]
                model_config_dict["blk" + str(i) + "fc1"] = model_config_list[i * 7 - 1]
                model_config_dict["blk" + str(i) + "fc2"] = model_config_list[i * 7]
            model_config_dict["head"] = model_config_list[-1]
        else:
            raise NotImplementedError("Other models are not supported yet")
        return model_config_dict

    def config_dict_to_model_format(self, model_config_dict: dict):
        """Convert the model_config_dict (intermediate for mixed-precision) to model_format (final format for the model)

        NOTE: model_config_dict is a dict instead of list, each key is the layer name, and the value is the mantissa bits for each layer
            The reason why we need this dict is for debugging... You can regard it as an intermediate

        Args:
            model_config_dict (dict): see the definition in the beginning of this file

        Returns:
            model_format: see the definition in the beginning of this file
        """
        if self.model_category == "deit":
            embed_format_dict = {
                "embed": layer_config_to_layer_format(model_config_dict["embed"])
            }
            head_format_dict = {
                "head": layer_config_to_layer_format(model_config_dict["head"])
            }
            attn_format_dicts = []
            mlp_format_dicts = []
            for i in range(1, self.blk_depth + 1):
                attn_format_dict = {
                    "qkv": layer_config_to_layer_format(
                        model_config_dict["blk" + str(i) + "qkv"]
                    ),
                    "mulqk": layer_config_to_layer_format(
                        model_config_dict["blk" + str(i) + "mulqk"]
                    ),
                    "sftm": layer_config_to_layer_format(
                        model_config_dict["blk" + str(i) + "sftm"]
                    ),
                    "mulsv": layer_config_to_layer_format(
                        model_config_dict["blk" + str(i) + "mulsv"]
                    ),
                    "proj": layer_config_to_layer_format(
                        model_config_dict["blk" + str(i) + "proj"]
                    ),
                }
                attn_format_dicts.append(attn_format_dict)
                mlp_format_dict = {
                    "fc1": layer_config_to_layer_format(
                        model_config_dict["blk" + str(i) + "fc1"]
                    ),
                    "fc2": layer_config_to_layer_format(
                        model_config_dict["blk" + str(i) + "fc2"]
                    ),
                }
                mlp_format_dicts.append(mlp_format_dict)
            return [
                embed_format_dict,
                attn_format_dicts,
                mlp_format_dicts,
                head_format_dict,
            ]
        else:
            raise NotImplementedError("Other models are not supported yet")
