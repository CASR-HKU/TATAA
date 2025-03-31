from parser.layer_parser import LayerParser
from parser.model_spec import ModelSpec
from parser.node_param import NodeParam
from parser.operation_param import OperationParam
import torch.nn.functional as F
import torch

bfp_ops = ['linear', 'matmul']
fp_ops = ['mul', 'sub', 'add', 'div']
nonlinear_layer_list = ['norm', 'sftm', 'gelu']
layer_type_with_weight_and_bias = ['HMQConv2d', 'HMQLinear', 'HMQLayerNorm']

def padShape(n, dim=32): 
    if n % dim == 0: 
        return n
    else:
        return n - (n % dim) + dim

class ModelParser:
    """Model parser.
    """
    def __init__(self, model_name, batch_size=0) -> None:
        self.ms_param = {'name': model_name, 
                    'input_size': (), 
                    'weight_size': [], # temp weight size list
                    'batch_size': batch_size,
                    'constants': {}
        }
        self.ms = ModelSpec(self.ms_param)
        self.bs = batch_size

    def parse_layer(self, layer_name, node_type='fp16') -> ModelSpec:
        """Parse the model layer and return a LayerSpec.
        """
        self.base_lp = LayerParser(layer_name, node_type)
        node_param_dict = self.base_lp.get_layer_param()
        node_param = NodeParam(node_param_dict)

        self.ms.nodes.append(node_param)
    
    def parse_ops(self, op_name='', 
                  in_data=None, in_data_name=None, 
                  in_tmp=None, in_tmp_name=None, 
                  out_data=None, out_data_name=None,
                  out_tmp=None, out_tmp_name=None, 
                  feature_dict=None, keep_dim=False, keep_out=False,
                  int_ops=False, verbose=False):
        """Parse the layer operations and return a OpsSpec
        """
        if in_data is not None:
            in_data_shape = in_data.shape
            if not keep_dim:
                if in_data_shape[0] != 0:
                    if len(in_data_shape) == 4:
                        in_data = in_data.flatten(0,1).transpose(-1,-2)
                        in_data_shape = in_data.shape
                    else:
                        in_data = in_data.transpose(-1,-2)
                        in_data_shape = in_data.shape
                    if in_data_shape[0] == self.bs:
                        in_data_shape = in_data_shape[1:]

        if in_tmp is not None:
            in_tmp_shape = in_tmp.shape
            if not keep_dim:
                if in_tmp_shape[0] != 0:
                    if len(in_tmp_shape) == 4:
                        in_tmp = in_tmp.flatten(0,1).transpose(-1,-2)
                        in_tmp_shape = in_tmp.shape
                    else:
                        in_tmp = in_tmp.transpose(-1,-2)
                        in_tmp_shape = in_tmp.shape
                    if in_tmp_shape[0] == self.bs:
                        in_tmp_shape = in_tmp_shape[1:]

        if out_data is not None:
            out_data_shape = out_data.shape
            if not keep_out:
                if out_data_shape[0] != 0:
                    if len(out_data_shape) == 4:
                        out_data = out_data.flatten(0,1).transpose(-1,-2)
                        out_data_shape = out_data.shape
                    else:
                        out_data = out_data.transpose(-1,-2)
                        out_data_shape = out_data.shape
                    if out_data_shape[0] == self.bs:
                        out_data_shape = out_data_shape[1:]

        if out_tmp is not None:
            out_tmp_shape = out_tmp.shape
            if not keep_out:
                if out_tmp_shape[0] != 0:
                    if len(out_tmp_shape) == 4:
                        out_tmp = out_tmp.flatten(0,1).transpose(-1,-2)
                        out_tmp_shape = out_tmp.shape
                    else:
                        out_tmp = out_tmp.transpose(-1,-2)
                        out_tmp_shape = out_tmp.shape
                    if out_tmp_shape[0] == self.bs:
                        out_tmp_shape = out_tmp_shape[1:]

        op_param_dict = self.base_lp.get_op_param(op_name)

        if in_data is not None:
            if len(in_data_shape) <= 3:
                op_param_dict.update({'xin': in_data_shape})
            else:
                op_param_dict.update({'xin': in_data_shape[1:]})
        if in_tmp is not None:
            if len(in_tmp_shape) <= 3:
                op_param_dict.update({'yin': in_tmp_shape})
            else:
                op_param_dict.update({'yin': in_tmp_shape[1:]})
        if out_data is not None:
            if len(out_data_shape) <= 3:
                op_param_dict.update({'zout': out_data_shape})
            else:
                op_param_dict.update({'zout': out_data_shape[1:]})
        if out_tmp is not None:
            if len(out_data_shape) <= 3:
                op_param_dict.update({'tout': out_tmp_shape})
            else:
                op_param_dict.update({'tout': out_tmp_shape[1:]})

        if not int_ops:
            if in_data_name is not None:  
                op_param_dict.update({"x_ref": in_data_name})
            if in_tmp_name is not None:
                op_param_dict.update({"y_ref": in_tmp_name})
            if out_data_name is not None:
                op_param_dict.update({"z_ref": out_data_name})
            if out_tmp_name is not None:
                op_param_dict.update({"t_ref": out_tmp_name})
        if feature_dict is not None:
            for key, value in feature_dict.items():
                op_param_dict.update({key: value})

        operation_param = OperationParam(op_param_dict)
        self.ms.nodes[-1].operations.append(operation_param)

    def return_ms(self):
        """Return the Model Spec.
        """
        return self.ms