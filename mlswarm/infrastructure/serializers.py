class ServiceSerializerMixin:
    service_cls = None

    def create(self, validated_data):
        return self.service_cls(**validated_data)

    def update(self, instance, validated_data):
        for field in self.fields:
            instance = validated_data.get(field, getattr(instance, field))

        return instance
