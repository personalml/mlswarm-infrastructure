import abc


class ITrainer(metaclass=abc.ABCMeta):
    def __init__(self, **data):
        self.__dict__ = data

    def train(self, d):
        raise NotImplementedError


class NetworkTrainer(ITrainer):
    def train(self, d):
        return {'trained': 'ok'}
