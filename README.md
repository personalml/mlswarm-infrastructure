# Solvian Machine Learning

Machine Learning infrastructure for Solvian services.

## Installing

```shell
cd ./solvian-ml/
python setup.py install
```

## Usage Example

### Parsers

A few parsers are implemented to convert json and csv files into
a pandas DataFrame, ideal to feed machine learning models with:

```python
from solvian.ml.infrastructure.parsers import JsonParser

parser = JsonParser(content='path/to/file.json',
                    ignore_features=['temperature'])
# or `parser = CsvParser(content='path/to/file.csv', ...)`

d = parser.process()

print('features:', d.columns)
print(d.head())

m = SomeMachineLearningModel(...)
print(m.predict(d))
```
