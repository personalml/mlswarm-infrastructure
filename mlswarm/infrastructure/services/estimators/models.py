import abc
import os
from typing import Dict

import numpy as np
from keras import Model, Input, callbacks, backend as K
from keras import models
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.base import TransformerMixin
from sklearn.externals import joblib
from sklearn.preprocessing import CategoricalEncoder, StandardScaler


class IEstimator(metaclass=abc.ABCMeta):
    def __init__(self, target=None):
        self.target = target

    def train(self, d, **params):
        raise NotImplementedError

    def test(self, d, **params):
        raise NotImplementedError

    def predict(self, d, **params):
        raise NotImplementedError

    def load(self, directory: str) -> 'IEstimator':
        raise NotImplementedError

    def save(self, directory: str) -> 'IEstimator':
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
        return self

    def load(self, directory: str):
        return self

    def save(self, directory: str):
        return self


class SimpleDenseNetworkClassifier(IEstimator):
    def __init__(self, input_units, output_units,
                 inner_units, inner_layers,
                 activations: str = 'relu',
                 target: str = 'label',
                 preprocessor: Dict[str, TransformerMixin] = None,
                 model: Model = None):
        super().__init__(target=target)
        self.input_units = input_units
        self.output_units = output_units
        self.inner_units = inner_units
        self.inner_layers = inner_layers
        self.activations = activations
        self.preprocessor = preprocessor
        self.model = model

    def build(self, dropout=0.0) -> Model:
        y = x = Input(shape=[self.input_units])

        for i in range(self.inner_layers):
            y = Dropout(rate=dropout, name='dr_%i' % i)(y)
            y = Dense(self.inner_units, use_bias=False, name='fc_%i' % i)(y)
            y = BatchNormalization(name='bn_%i' % i)(y)
            y = Activation(self.activations, name='ac_%i' % i)(y)

        y = Dense(self.output_units, activation='softmax', name='predictions')(y)
        self.model = Model(inputs=x, outputs=y)

        return self.model

    def train(self, d, report_dir=None, dropout=0.5, batch_size=32,
              epochs=5, validation_split=0., **params):
        d = d.sample(frac=1)
        x = d[[c for c in d.columns if c != self.target]]
        y = d[[self.target]]

        self.preprocessor = dict(
            x=StandardScaler(),
            y=CategoricalEncoder(encoding='onehot-dense'),
        )

        x = self.preprocessor['x'].fit_transform(x)
        y = self.preprocessor['y'].fit_transform(y)

        self.build(dropout=dropout)

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        callbacks_used = [callbacks.TerminateOnNaN()]

        if validation_split:
            callbacks_used += [
                callbacks.TensorBoard(report_dir, batch_size=batch_size,
                                      histogram_freq=1, write_grads=True),
                callbacks.ModelCheckpoint(os.path.join(report_dir, 'network.h5'),
                                          verbose=0, save_best_only=True)
            ]

        report = self.model.fit(x, y,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=0,
                                callbacks=callbacks_used,
                                validation_split=validation_split,
                                validation_data=None,
                                shuffle=True,
                                class_weight=None,
                                sample_weight=None,
                                initial_epoch=0,
                                steps_per_epoch=None,
                                validation_steps=None)

        if not validation_split:
            models.save_model(self.model, os.path.join(report_dir, 'network.h5'))

        return {k: [float(_v) for _v in v] for k, v in report.history.items()}

    def test(self, d, batch_size=32, **params):
        x = d[[c for c in d.columns if c != self.target]]
        y = d[[self.target]]
        x = self.preprocessor['x'].transform(x)
        y = self.preprocessor['y'].transform(y)

        losses = self.model.evaluate(x, y, batch_size=batch_size, verbose=0)
        if not isinstance(losses, (list, np.ndarray)):
            losses = [losses]

        return dict(zip(self.model.metrics_names, losses))

    def predict(self, d, batch_size=32, **params):
        x = d[[c for c in d.columns if c != self.target]]
        x = self.preprocessor['x'].transform(x)

        return (self.model
                .predict(x, batch_size=batch_size, verbose=0)
                .argmax(axis=-1)
                .tolist())

    def save(self, directory: str):
        joblib.dump(self.preprocessor, os.path.join(directory, 'processor.p'))
        return self

    def load(self, directory: str):
        if self.model is None:
            self.build()

        self.preprocessor = joblib.load(os.path.join(directory, 'processor.p'))
        self.model = models.load_model(os.path.join(directory, 'network.h5'))
        return self

    def dispose(self):
        self.preprocessor = self.model = None

        return self
