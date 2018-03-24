import abc
import itertools
import json
import os
from io import StringIO

import numpy as np
import pandas as pd


class IParser(metaclass=abc.ABCMeta):
    def __init__(self, content, to_lowercase=False, ignore_features=None):
        self.content = content
        self.to_lowercase = to_lowercase
        self.ignore_features = ignore_features

    def parse(self):
        """Parse the data, reading it for the estimators.
        """
        raise NotImplementedError

    def concatenate(self, parsers):
        raise NotImplementedError


class CSVParser(IParser):
    def __init__(self, *args, delimiter=',', **kwargs):
        super().__init__(*args, **kwargs)
        self.delimiter = delimiter

    def parse(self):
        try:
            # Read from path or url.
            d = pd.read_csv(self.content, delimiter=self.delimiter)
        except FileNotFoundError:
            # Maybe it's a csv string?
            d = pd.read_csv(StringIO(self.content), delimiter=self.delimiter)

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

    def concatenate(self, parsers):
        return pd.concat([p.parse() for p in parsers])


class JSONParser(IParser):
    def parse(self):
        if os.path.exists(self.content):
            with open(self.content) as f:
                d = json.load(f)
        else:
            d = json.loads(self.content)

        return pd.read_csv(StringIO(d))

    def concatenate(self, parsers):
        d = [p.parse() for p in parsers]
        d = list(itertools.chain.from_iterable(p if isinstance(p, list)
                                               else [p]
                                               for p in d))
        return d
