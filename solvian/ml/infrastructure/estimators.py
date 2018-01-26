import abc


class IEstimator(metaclass=abc.ABCMeta):
    def __init__(self,
                 input_units,
                 output_units,
                 target='Label',
                 trained=False):
        self.input_units = input_units
        self.output_units = output_units
        self.target = target

        self.trained_ = trained


class INetworkEstimator(IEstimator, metaclass=abc.ABCMeta):
    def __init__(self,
                 input_units,
                 output_units,
                 inner_units=None,
                 inner_layers=2,
                 activations='relu',
                 target='Label',
                 trained=False):
        super().__init__(input_units, output_units, target, trained)
        self.input_units = input_units
        self.output_units = output_units
        self.inner_layers = inner_layers
        self.inner_units = inner_units
        self.activations = activations
