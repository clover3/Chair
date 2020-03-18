def lmap(func, iterable_something):
    return list([func(e) for e in iterable_something])


def lmap_w_exception(func, iterable_something, exception):
    class Fail:
        pass

    def func_warp(e):
        try:
            return func(e)
        except exception:
            return Fail()
    r1 = list([func_warp(e) for e in iterable_something])
    return list([e for e in r1 if type(e) != Fail])


def l_to_map(func, iterable_something):
    return {k: func(k) for k in iterable_something}


def idx_where(func, iterable_something):
    return [idx for idx, item in enumerate(iterable_something) if func(item)]


def dict_map(func, dict_like):
    return {k: func(v) for k,v in dict_like.items()}


def lfilter(func, iterable_something):
    return list(filter(func, iterable_something))


def foreach(func, iterable_something):
    for e in iterable_something:
        func(e)


def reverse(l):
    return list(reversed(l))


def flatten(z):
    return [y for x in z for y in x]


def left(pairs):
    return list([a for a,b in pairs])


def right(pairs):
    return list([b for a,b in pairs])