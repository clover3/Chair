import os.path
import sys
import time
from typing import List, Iterable, Callable, Dict, Tuple, Set


# Return true if succesfully terminate
def wait_trigger(
        trigger_fn: Callable[[], bool],
        check_interval=60,
        max_wait=1000 * 1000 * 1000,
        print_waiting_time=True,
) -> bool:
    acc_sleep_time = 0
    terminate = trigger_fn()
    while not terminate and acc_sleep_time < max_wait:
        acc_sleep_time += check_interval
        if print_waiting_time:
            print("\r Sleeping {} mins".format(int(acc_sleep_time / 60)), end="")
        time.sleep(check_interval)
        terminate: bool = trigger_fn()

    return terminate


def wait_file(file_path):
    def trigger_fn():
        return os.path.exists(file_path)

    print("Wait for the model: ", file_path)
    return wait_trigger(trigger_fn)


if __name__ == "__main__":
    wait_file(sys.argv[1])
