from .. import errors


class ServiceBuilder:
    def __init__(self, classes: dict):
        self.classes = classes

    def get(self, _id: str):
        try:
            return self.classes[_id]
        except KeyError:
            raise errors.ClassNotFoundInBagError(
                'Cannot find class {%s} in this bag. Available options are: %s.'
                % (_id, self.registered))

    def build(self, _id: str, *args, **kwargs):
        return self.get(_id)(*args, **kwargs)

    @property
    def registered(self):
        return list(self.classes.keys())

    def to_choices(self):
        choices = self.registered
        return list(zip(choices, choices))
