import abc


class IEstimator(metaclass=abc.ABCMeta):
    def __init__(self, **data):
        self.__dict__ = data

    def train(self, d):
        raise NotImplementedError

    def test(self, d):
        raise NotImplementedError

    def predict(self, d):
        raise NotImplementedError


class SimpleDenseNetworkClassifier(IEstimator):
    def train(self, d):
        return {'trained': 'ok'}

    def test(self, d):
        return {'tested': 'ok'}

    def predict(self, d):
        return {'predicted': 'ok'}


class SimpleRegressor(IEstimator):
    def train(self, d):
        return {'trained': 'ok'}

    def test(self, d):
        return {'tested': 'ok'}

    def predict(self, d):
        return {'predicted': 'ok'}
