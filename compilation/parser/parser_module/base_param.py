import json
import re

class MyJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, BaseParam):
            return dict(o)
        else:
            return super().default(o)

class BaseParam:
    """Base class for other param classes.
    """

    def __init__(self, params: dict) -> None:
        """Initialize BaseParam.

        Arguments:
            params: A dictionary of parameters.
        """
        self.params = params

    def __iter__(self) -> iter:
        for k in self.get_param_keys(self):
            yield k, getattr(self, k)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            str_tmp = json.dumps(self, indent=4, cls=MyJSONEncoder)
            # remove indent in list[int]
            str_tmp = re.sub(r"\n\s*(\d+,?)", r"\g<1> ", str_tmp)
            str_tmp = re.sub(r"(\d+)\s\n\s+\]", r"\g<1>]", str_tmp)
            f.write(str_tmp)

    @staticmethod
    def get_param_keys(self) -> tuple[str]:
        """A tuple of required keys.
        """
        raise NotImplementedError

    @property
    def params(self) -> dict:
        """Param dict.
        """
        return self._params
    @params.setter
    def params(self, params: dict) -> None:
        # print("Base Param Key: ", self.get_param_keys(self))
        for k in self.get_param_keys(self):
            # print("--> k: ", k)
            if k in params:
                setattr(self, f"_{k}", params[k])
            else:
                raise KeyError(f"{k} is not in params: {params}")
        self._params = params