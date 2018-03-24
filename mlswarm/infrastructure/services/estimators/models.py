import abc

from sklearn.preprocessing import CategoricalEncoder
from keras import Model, Input, callbacks, backend as K
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical


class IEstimator(metaclass=abc.ABCMeta):
    def __init__(self, target=None):
        self.target = target

    def train(self, d, **params):
        raise NotImplementedError

    def test(self, d, **params):
        raise NotImplementedError

    def predict(self, d, **params):
        raise NotImplementedError

    def dispose(self):
        raise NotImplementedError


class DummyRegressor(IEstimator):
    def train(self, d, **params):
        return {'trained': 'ok'}

    def test(self, d, **params):
        return {'tested': 'ok'}

    def predict(self, d, **params):
        return {'predicted': 'ok'}

    def dispose(self):
        pass


class SimpleDenseNetworkClassifier(IEstimator):
    def __init__(self, input_units, output_units,
                 inner_units, inner_layers,
                 activations='relu',
                 target='label'):
        super().__init__(target=target)
        self.input_units = input_units
        self.output_units = output_units
        self.inner_units = inner_units
        self.inner_layers = inner_layers
        self.activations = activations
        self.model_ = None

    def build(self, dropout=0.0):
        y = x = Input(shape=[self.input_units])

        for i in range(self.inner_layers):
            y = Dropout(rate=dropout, name='dr_%i' % i)(y)
            y = Dense(self.inner_units, use_bias=False, name='fc_%i' % i)(y)
            y = BatchNormalization(name='bn_%i' % i)(y)
            y = Activation(self.activations, name='ac_%i' % i)(y)

        y = Dense(self.output_units, activation='softmax', name='predictions')(y)
        self.model_ = Model(inputs=x, outputs=y)

    def dispose(self):
        if self.model_ is not None:
            self.model_ = None
            K.clear_session()

    def train(self, d, report_dir=None, dropout=0.5, batch_size=32, **params):
        # is_categorical = d.dtypes == object
        # CategoricalEncoder
        x = d[[c for c in d.columns if c != self.target]]
        y = d[self.target]
        y = to_categorical(y)

        self.build(dropout=dropout)

        report = self.model_.fit(x, y,
                                 batch_size=batch_size,
                                 epochs=params.get('epochs'),
                                 verbose=2,
                                 callbacks=[
                                     callbacks.TerminateOnNaN(),
                                     callbacks.TensorBoard(report_dir, batch_size=batch_size),
                                     callbacks.ModelCheckpoint(report_dir, verbose=1, save_best_only=True)
                                 ],
                                 validation_split=params.get('validation_split'),
                                 validation_data=None,
                                 shuffle=True,
                                 class_weight=None,
                                 sample_weight=None,
                                 initial_epoch=0,
                                 steps_per_epoch=None,
                                 validation_steps=None)

        return {k: [float(_v) for _v in v] for k, v in report.history.items()}

    def test(self, d, **params):
        return {'tested': 'ok'}

    def predict(self, d, **params):
        return {'predicted': 'ok'}
