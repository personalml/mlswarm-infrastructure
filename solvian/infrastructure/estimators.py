import abc


class IEstimator(metaclass=abc.ABCMeta):
    def __init__(self, input_units, inner_units, output_units, inner_layers,
                 activations, target, **kwargs):
        self.input_units = input_units
        self.output_units = output_units
        self.inner_layers = inner_layers
        self.units = inner_units
        self.activations = activations
        self.target = target
        self._kwargs = kwargs

    def train(self, data, train_params):
        raise NotImplementedError

    def test(self, data, test_params):
        raise NotImplementedError

    def predict(self, data, predict_params):
        raise NotImplementedError
