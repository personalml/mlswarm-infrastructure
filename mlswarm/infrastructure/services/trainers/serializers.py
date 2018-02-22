from rest_framework import serializers
from rest_framework.serializers import Serializer

from . import models
from ...serializers import ServiceSerializerMixin


class ITrainer(ServiceSerializerMixin,
               Serializer):
    pass


class NetworkTrainer(ITrainer):
    learning_rate = serializers.FloatField(
        min_value=0,
        default=0.01,
        help_text='Learning rate used by the optimizer when adjusting weights.')
    optimizer = serializers.CharField(
        max_length=32,
        default='adam',
        help_text='Optimizer used to train the network.')
    dropout = serializers.FloatField(
        min_value=0,
        max_value=1,
        default=0,
        help_text='Dropout probability rate used in the dropout layers. 0 means no dropout at all.')

    service_cls = models.NetworkTrainer
