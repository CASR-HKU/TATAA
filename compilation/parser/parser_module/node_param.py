from parser.base_param import BaseParam
from parser.operation_param import OperationParam

class NodeParam(BaseParam):

    def __init__(self, params: dict) -> None:
        """Initialize NodeParam.
        """
        super().__init__(params)

    @staticmethod
    def get_param_keys(self) -> tuple[str]:
        """A tuple of dictionary keys. All required when save.
        """
        return ('name', 'node_type', 'operations')

    @BaseParam.params.setter
    def params(self, params: dict) -> None:
        new_params = {}
        new_params.update(params)
        new_params.setdefault('node_type')
        new_params.setdefault('operations', [])
        for idx, operation in enumerate(new_params['operations']):
            new_params['operations'][idx] = (operation if isinstance(operation, OperationParam)
                                        else OperationParam(operation))
        BaseParam.params.fset(self, new_params)

    @property
    def name(self) -> str:
        """ Name of the node.
        """
        return self._name
    
    @property
    def node_type(self) -> str:
        """ layer_type of the node.
        """
        return self._node_type

    @property
    def operations(self) -> list[OperationParam]:
        """A list of OperationParam.
        """
        return self._operations