from parser.base_param import BaseParam
from parser.node_param import NodeParam

class ModelSpec(BaseParam):

    @staticmethod
    def get_param_keys(self) -> tuple[str]:
        """A tuple of dictionary keys. All required when save.
        """
        return (
            'name',
            'input_size',
            'weight_size',
            'batch_size',
            'constants',
            'nodes',
        )

    @BaseParam.params.setter
    def params(self, params: dict) -> None:
        new_params = {}
        new_params.update(params)
        new_params.setdefault('nodes', [])
        for idx, node in enumerate(new_params['nodes']):
            new_params['nodes'][idx] = (node if isinstance(node, NodeParam)
                                        else NodeParam(node))
        BaseParam.params.fset(self, new_params)

    def update_param(self, key: str, value) -> None:
        """Update a parameter.
        """
        if key in self.get_param_keys(self):
            setattr(self, f"_{key}", value)
        else:
            raise KeyError(f"{key} is not in params: {self.params}")

    @property
    def name(self) -> str:
        """Model name.
        """
        return self._name
    
    @property
    def input_size(self) -> tuple:
        """Input image size.
        """
        return self._input_size

    @property
    def batch_size(self) -> tuple:
        """Batch size.
        """
        return self._batch_size
    
    @property
    def weight_size(self) -> tuple:
        """Weight size.
        """
        return self._weight_size

    @property
    def constants(self) -> tuple:
        """Constant value.
        """
        return self._constants

    @property
    def nodes(self) -> list[NodeParam]:
        """A list of NodeParam.
        """
        return self._nodes
    
    
