from typing import List, Tuple

from list_lib import lmap
from trainer.promise import PromiseKeeper, MyFuture, MyPromise, list_future

A = int

class C:
    def __init__(self, v:int):
        self.v = v

class B:
    def __init__(self, v: int):
        self.v = v

    def __str__(self):
        return "B({})".format(self.v)

class D:
    def __init__(self, v: int):
        self.v = v


class E:
    def __init__(self, v: str):
        self.v = v

    def __str__(self):
        return "E({})".format(self.v)


# Each A makes two B
# Each B makes multiple C
# Each C is converted D  ## This should be done in batch
# multiple D reduces to E
# Output: E, E

# A_List

def my_future_example_main():
    a_list: List[A] = list(range(0, 100, 10))

    def a_to_bb(a: A) -> Tuple[B, B]:
        bb = B(a-1), B(a+1)
        return bb

    bb_list: List[Tuple[B, B]] = list(map(a_to_bb, a_list))

    def b_to_c_list(b: B) -> List[C]:
        return [C(i) for i in range(b.v, b.v+10)]

    def c_list_to_d_list(c_list: List[C]) -> List[D]:
        return [D(c.v) for c in c_list]

    def d_list_to_e(d_list: List[D]) -> E:
        return E(" ".join(str(d.v) for d in d_list))

    ## This is what we have

    def item_pk(func):
        return PromiseKeeper(lambda k_list: [func(k) for k in k_list])

    def b_to_e_mapper(b_list: List[B]) -> List[E]:
        pk_c_to_d = PromiseKeeper(c_list_to_d_list)
        pk_dl_to_e = item_pk(d_list_to_e)
        c_list_list: List[List[C]] = lmap(b_to_c_list, b_list)

        def c_list_mapper(c_list) -> List[MyFuture[D]]:
            return [MyPromise(c, pk_c_to_d).future() for c in c_list]

        pk_lf_to_f = item_pk(list_future)

        def fl_to_f(f_list: List[MyFuture]) -> MyFuture[List]:
            assert type(f_list) == list
            assert type(f_list[0]) == MyFuture
            promise: MyPromise = MyPromise(f_list, pk_lf_to_f)
            return promise.future()

        future_d_list_list: List[List[MyFuture[D]]] = lmap(c_list_mapper, c_list_list)
        # list_future_map(c_list_list, c_list)
        fld_list: List[MyFuture[List[D]]] = lmap(fl_to_f, future_d_list_list)
        pk_c_to_d.do_duty()
        pk_lf_to_f.do_duty()
        future_e_list: List[MyFuture[E]] = [MyPromise(fld.get(), pk_dl_to_e).future() for fld in fld_list]
        pk_dl_to_e.do_duty()
        return list_future(future_e_list)

    pk_b_to_e = PromiseKeeper(b_to_e_mapper)
    def b_to_e_future(b: B) -> MyFuture[E]:
        return MyPromise(b, pk_b_to_e).future()

    def f_tuple(fp: Tuple[MyFuture, MyFuture]) -> Tuple:
        f1, f2 = fp
        t1, t2 = f1.get(), f2.get()
        return t1, t2

    pk_f_tuple = item_pk(f_tuple)

    def fp_to_f(fp: Tuple[MyFuture, MyFuture]) -> MyFuture[Tuple]:
        promise: MyPromise = MyPromise(fp, pk_f_tuple)
        return promise.future()

    ltfe: List[Tuple[MyFuture[E], MyFuture[E]]] = [(b_to_e_future(b1), b_to_e_future(b2)) for (b1, b2) in bb_list]
    lfte: List[MyFuture[Tuple[E, E]]] = list(map(fp_to_f, ltfe))
    pk_b_to_e.do_duty()
    pk_f_tuple.do_duty()

    lte: List[Tuple[E, E]] = list_future(lfte)
    for te in lte:
        e1, e2 = te
        print(e1, e2)


if __name__ == "__main__":
    my_future_example_main()