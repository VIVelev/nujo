''' this decorators are ment to be used with
line/memory profiler's @profile decorator
'''

__all__ = [
    'decorate_if',
    'decorate_if_defined',
]


def decorate_if(condition, decorator):
    return decorator if condition else lambda x: x


def decorate_if_defined(decorator):
    return globals().get(decorator, lambda x: x)
