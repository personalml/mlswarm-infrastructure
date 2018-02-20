class SwarmError(Exception):
    pass


class MalformedEstimator(SwarmError):
    pass


class SwarmApiUserError(SwarmError, RuntimeError):
    pass


class ClassNotFoundInBagError(SwarmApiUserError):
    pass


class NotFittedError(SwarmApiUserError):
    pass
