import time
from concurrent.futures import ThreadPoolExecutor


def func1(item):
    time.sleep(2)
    return item * 2


a_list = list(range(10))


with ThreadPoolExecutor(max_workers=1) as executor:
    def get_future(item):
        return executor.submit(func1, item)

    future_list = [get_future(t) for t in a_list]
    print(future.done())
    print(future.result())


