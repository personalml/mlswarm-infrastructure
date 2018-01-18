import abc

import numpy as np
import pandas as pd


class IParser(metaclass=abc.ABCMeta):
    def __init__(self,
                 content,
                 delimiter=',',
                 to_lowercase=False,
                 ignore_features=(),
                 **kwargs):
        self.content = content
        self.delimiter = delimiter
        self.to_lowercase = to_lowercase
        self.ignore_features = ignore_features
        self._kwargs = kwargs

    def parse(self):
        """Parse the data, reading it for the estimators.
        """
        raise NotImplementedError

    def process(self):
        d = self.parse()

        if self.ignore_features:
            retained = [c for c in d.columns if c not in self.ignore_features]
            d = d[retained]

        if self.to_lowercase:
            # Lower case everything.
            categoricals = d.columns[d.dtypes == np.object_]
            d[categoricals] = d[categoricals].str.lower()

        return d


class CsvDatasetParser(IParser):
    def parse(self):
        return pd.read_csv(self.content, delimiter=self.delimiter)
