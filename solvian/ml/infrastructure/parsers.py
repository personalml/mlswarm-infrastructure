import abc
import json
import os
from io import StringIO

import numpy as np
import pandas as pd
import pandas.io.json


class IParser(metaclass=abc.ABCMeta):
    def __init__(self,
                 content,
                 to_lowercase=False,
                 ignore_features=()):
        self.content = content
        self.to_lowercase = to_lowercase
        self.ignore_features = ignore_features

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
            # Lower case everything, but enforce str first (some
            # categorical columns have strings and numbers mixed).
            categoricals = d.columns[d.dtypes == np.object_]
            for c in categoricals:
                d[c] = d[c].apply(str).str.lower()

        return d


class CsvParser(IParser):
    def __init__(self,
                 content,
                 to_lowercase=False,
                 ignore_features=(),
                 delimiter=','):
        super().__init__(content, to_lowercase, ignore_features)
        self.delimiter = delimiter

    def parse(self):
        try:
            # Read from path or url.
            return pd.read_csv(self.content, delimiter=self.delimiter)
        except FileNotFoundError:
            # Maybe it's a csv string?
            return pd.read_csv(StringIO(self.content), delimiter=self.delimiter)


class JsonParser(IParser):
    def __init__(self,
                 content,
                 to_lowercase=False,
                 ignore_features=(),
                 normalize=False,
                 record_path=None,
                 meta=None):
        super().__init__(content,
                         to_lowercase=to_lowercase,
                         ignore_features=ignore_features)
        self.normalize = normalize
        self.record_path = record_path
        self.meta = meta

    def parse(self):
        if os.path.exists(self.content):
            with open(self.content) as f:
                d = json.load(f)
        else:
            d = json.loads(self.content)

        if self.normalize:
            return pd.io.json.json_normalize(d,
                                             record_path=self.record_path,
                                             meta=self.meta)

        return pd.read_csv(StringIO(d))
