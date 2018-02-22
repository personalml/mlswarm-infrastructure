from rest_framework import serializers
from rest_framework.serializers import Serializer

from . import models
from ...serializers import ServiceSerializerMixin


class IEstimator(ServiceSerializerMixin,
                 Serializer):
    target = serializers.CharField(required=False)


class SimpleDenseNetworkClassifier(IEstimator):
    input_units = serializers.IntegerField(min_value=1, required=True)
    inner_units = serializers.IntegerField(min_value=1, required=True)
    output_units = serializers.IntegerField(min_value=1, required=True)
    inner_layers = serializers.IntegerField(min_value=0, required=True)
    activations = serializers.CharField(max_length=32, default='relu')

    service_cls = models.SimpleDenseNetworkClassifier


class SimpleRegressor(IEstimator):
    input_units = serializers.IntegerField(min_value=1, required=True)
    output_units = serializers.IntegerField(min_value=1, required=True)

    service_cls = models.SimpleRegressor
