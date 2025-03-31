class LayerParser:
    """Layer parser.
    """

    def __init__(self, layer_name, node_type) -> None:
        self._layer_name = layer_name
        self._node_type = node_type

    def get_layer_param(self) -> dict:
        """Return the dict of layer parameters.
        """
        param_dict = {}   
        param_dict['name'] = self._layer_name
        param_dict['node_type'] = self._node_type
        return param_dict
    
    def get_op_param(self, op_name) -> dict:
        """Return the dict of operation parameters.
        """
        param_dict = {}   
        param_dict['op_name'] = op_name
        return param_dict