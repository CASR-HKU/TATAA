class BitType:

    def __init__(self, bits, signed, name=None):
        self.bits = bits
        self.signed = signed
        if name is not None:
            self.name = name
        else:
            self.update_name()

    @property
    def upper_bound(self):
        if not self.signed:
            return 2**self.bits - 1
        return 2 ** (self.bits - 1) - 1

    @property
    def lower_bound(self):
        if not self.signed:
            return 0
        return -(2 ** (self.bits - 1))

    @property
    def range(self):
        return 2**self.bits

    def update_name(self):
        self.name = ""
        if not self.signed:
            self.name += "uint"
        else:
            self.name += "int"
        self.name += "{}".format(self.bits)


class Config:

    def __init__(self):

        self.BIT_TYPE_W = BitType(8, True, "int8")
        self.BIT_TYPE_A = BitType(8, True, "int8")

        self.OBSERVER_W = "minmax"
        self.OBSERVER_A = "minmax"

        self.QUANTIZER_W = "uniform"
        self.QUANTIZER_A = "uniform"

        self.CALIBRATION_MODE_W = "layer_wise"
        self.CALIBRATION_MODE_A = "layer_wise"
