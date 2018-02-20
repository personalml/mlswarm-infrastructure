import abc

from marshmallow import Schema

from . import schemas, errors


class IEstimator(metaclass=abc.ABCMeta):
    schema = schemas.Estimator()

    def __init__(self, **data):
        if self.schema is None or not isinstance(self.schema, Schema):
            raise errors.MalformedEstimator('%s does not have a valid '
                                            'marshmallow schema: %s'
                                            % (self, self.schema))
        self.data = self.schema.load(data)
