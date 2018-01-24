import abc


class IEstimator(metaclass=abc.ABCMeta):
    def __init__(self,
                 input_units,
                 inner_units,
                 output_units,
                 inner_layers=2,
                 activations='relu',
                 target='Label',
                 trained=False,
                 **kwargs):
        self.input_units = input_units
        self.output_units = output_units
        self.inner_layers = inner_layers
        self.units = inner_units
        self.activations = activations
        self.target = target
        self._kwargs = kwargs

        self.trained_ = trained

    def train(self, d, *args, **params):
        raise NotImplementedError

    def test(self, d, *args, **params):
        raise NotImplementedError

    def predict(self, d, *args, **params):
        raise NotImplementedError
