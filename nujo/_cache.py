__all__ = [
    'cached_property',
]


class cached_property:
    ''' A property that is only computed once per instance and then replaces itself
    with an ordinary attribute.

    Deleting the attribute resets the property.

    '''
    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self

        key = self.func.__name__
        cache = obj.__dict__

        if key in cache:
            return cache[key]
        else:
            value = cache[key] = self.func(obj)
            return value
