from parser.base_param import BaseParam

class OperationParam(BaseParam):

    def __init__(self, params: dict) -> None:
        """Initialize OperationParam.
        """
        self.operation_list = ()
        for k in params:
            self.operation_list += (k,)
            setattr(self, k, params[k])

    @staticmethod
    def get_param_keys(self) -> tuple[str]:
        """A tuple of dictionary keys. All required when save.
        """
        return self.operation_list

    @BaseParam.params.setter
    def params(self, params: dict) -> None:
        new_params = {}
        new_params.update(params)
        BaseParam.params.fset(self, new_params)

    def set_op_name(self, op_name: bool) -> None:
        """set operation name
        """
        self._op_name = op_name