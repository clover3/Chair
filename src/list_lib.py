from typing import Callable, TypeVar, Iterable, List, Dict, Tuple

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def lmap(func: Callable[[A], B],
         iterable_something: Iterable[A]) -> List[B]:
    return list([func(e) for e in iterable_something])


def lmap_w_exception(func: Callable[[A], B],
                     iterable_something: Iterable[A],
                     exception) -> List[B]:
    class Fail:
        pass

    def func_warp(e):
        try:
            return func(e)
        except exception:
            return Fail()
    r1 = list([func_warp(e) for e in iterable_something])
    return list([e for e in r1 if type(e) != Fail])


def l_to_map(func: Callable[[A], B],
         iterable_something: Iterable[A]) -> Dict[A, B]:
    return {k: func(k) for k in iterable_something}


def idx_where(func: Callable[[A], B],
         iterable_something: Iterable[A]) -> List[int]:
    return [idx for idx, item in enumerate(iterable_something) if func(item)]


def dict_value_map(func: Callable[[A], B], dict_like: Dict[C, A]) -> Dict[C, B]:
    return {k: func(v) for k, v in dict_like.items()}


def dict_key_map(func: Callable[[A], B], dict_like: Dict[A, C]) -> Dict[B, C]:
    return {func(k): v for k, v in dict_like.items()}


def lfilter(func: Callable[[A], B], iterable_something: Iterable[A]) -> List[A]:
    return list(filter(func, iterable_something))


def foreach(func, iterable_something):
    for e in iterable_something:
        func(e)


def reverse(l: Iterable[A]) -> List[A]:
    return list(reversed(l))


def flatten(z: Iterable[Iterable[A]]) -> Iterable[A]:
    return [y for x in z for y in x]


def left(pairs: Iterable[Tuple[A, B]]) -> List[A]:
    return list([a for a, b in pairs])


def right(pairs: Iterable[Tuple[A, B]]) -> List[B]:
    return list([b for a, b in pairs])
