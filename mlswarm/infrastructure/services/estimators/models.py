import abc


class IEstimator(metaclass=abc.ABCMeta):
    def __init__(self, **data):
        self.__dict__ = data

    def train(self, d, **params):
        raise NotImplementedError

    def test(self, d, **params):
        raise NotImplementedError

    def predict(self, d, **params):
        raise NotImplementedError


class SimpleDenseNetworkClassifier(IEstimator):
    def train(self, d, **params):
        return {'trained': 'ok'}

    def test(self, d, **params):
        return {'tested': 'ok'}

    def predict(self, d, **params):
        return {'predicted': 'ok'}


class SimpleRegressor(IEstimator):
    def train(self, d, **params):
        return {'trained': 'ok'}

    def test(self, d, **params):
        return {'tested': 'ok'}

    def predict(self, d, **params):
        return {'predicted': 'ok'}
