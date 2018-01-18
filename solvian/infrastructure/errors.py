class SolvianError(Exception):
    pass


class SolvianMlApiUserError(SolvianError, RuntimeError):
    pass


class ClassNotFoundInBagError(SolvianMlApiUserError):
    pass
