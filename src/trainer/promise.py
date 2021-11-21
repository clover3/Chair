import time
from typing import List, Callable, Generic
from typing import TypeVar

A = TypeVar('A')
B = TypeVar('B')


class PromiseKeeper:
    def __init__(self, list_fn: Callable[[A], B]):
        self.X_list = []
        self.list_fn = list_fn

    def do_duty(self, log_size=False):
        x_list = list([X.X for X in self.X_list])
        if log_size:
            print("{} items".format(len(x_list)))

        y_list = self.list_fn(x_list)
        for X, y in zip(self.X_list, y_list):
            X.future().Y = y


T = TypeVar('T')


class MyFuture(Generic[T]):
    def __init__(self):
        self.Y = None

    def get(self):
        if self.Y is None:
            raise Exception("Future is not ready.")
        return self.Y


class MyPromise:
    def __init__(self, X, promise_keeper: PromiseKeeper):
        self.X = X
        self.Y = MyFuture()
        promise_keeper.X_list.append(self)

    def future(self):
        return self.Y


def sum_future(futures):
    return sum([f.get() for f in futures])


def max_future(futures):
    return max([f.get() for f in futures])


# get from all element futures in the list
def list_future(futures):
    return list([f.get() for f in futures])


def promise_to_items(promises: List[MyPromise]):
    return list_future([p.future() for p in promises])



if __name__ == '__main__':
    def list_fn(l):
        r = []
        for i in l:
            r.append(i*2)
            time.sleep(1)
        return r

    pk = PromiseKeeper(list_fn)
    X_list = list(range(10))
    y_list = []
    for x in X_list:
        y = MyPromise(x, pk).future()
        y_list.append(y)

    pk.do_duty()

    for e in y_list:
        print(e.Y)
