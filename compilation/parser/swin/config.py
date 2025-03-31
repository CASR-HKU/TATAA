from bit_type import BitType

class Config:

    def __init__(self):
        self.BIT_TYPE_W = BitType(8, True, 'int8')
        self.BIT_TYPE_A = BitType(8, True, 'int8')

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = 'minmax'

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'

        self.BIT_TYPE_S = BitType(8, True, 'int8')
        self.OBSERVER_S = self.OBSERVER_A
        self.QUANTIZER_S = self.QUANTIZER_A
        
        self.OBSERVER_A_LN = self.OBSERVER_A
        self.CALIBRATION_MODE_A_LN = 'channel_wise'
