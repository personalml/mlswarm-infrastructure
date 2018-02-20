from marshmallow import Schema, fields


class Estimator(Schema):
    target = fields.String(null=False)
