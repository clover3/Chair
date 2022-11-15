from typing import List

from trainer.promise import MyFuture


class PromiseKeeperOnFuture:
    def __init__(self, single_fn, time_estimator=None):
        self.single_fn = single_fn
        self.promise_list: List[MyPromiseEx] = []

    def start_thread(self):
        # Run thread that checks
        pass

    def _thread(self):
        n_done = 0
        while n_done != len(self.promise_list):
            for p in self.promise_list:
                if p.x_future.f_ready and not p.future().f_ready:
                    y = self.single_fn(p.x_future.get())
                    p.future().set_value(y)

                if p.future().f_ready:
                    n_done += 1

    def get_future(self, x):
        return MyPromiseEx(x, self).future()


class MyPromiseEx:
    def __init__(self, x_future: MyFuture, promise_keeper: PromiseKeeperOnFuture):
        self.x_future = x_future
        self.Y = MyFuture()
        promise_keeper.promise_list.append(self)

    def future(self):
        return self.Y
