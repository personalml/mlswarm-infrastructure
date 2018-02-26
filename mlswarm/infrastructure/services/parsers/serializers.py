from rest_framework import serializers

from .models import CSVParser, JSONParser
from ...serializers import ServiceSerializerMixin


class BaseParserSerializer(ServiceSerializerMixin,
                           serializers.Serializer):
    content = serializers.CharField(
        help_text='The data or a valid path to it.')
    to_lowercase = serializers.BooleanField(
        default=False,
        help_text='Lowercase all strings in {content}.')
    ignore_features = serializers.ListField(
        default=(),
        help_text='Features in content to ignore, separated by the {delimiter}.')


class CSVParserSerializer(BaseParserSerializer):
    service_cls = CSVParser

    delimiter = serializers.CharField(
        max_length=1,
        default=',',
        help_text='The {content} delimiter.')


class JSONParserSerializer(BaseParserSerializer):
    service_cls = JSONParser

    normalize = serializers.BooleanField(default=False)
    record_path = serializers.CharField(default=None, allow_null=True)
    meta = serializers.CharField(default=None, allow_null=True)
