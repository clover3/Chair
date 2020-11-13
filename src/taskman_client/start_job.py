import os
import sys

from taskman_client.task_proxy import get_task_proxy


def main():
    if 'uuid' in os.environ:
        uuid_var = os.environ['uuid']
    else:
        uuid_var = None
    run_name = sys.argv[1]
    proxy = get_task_proxy(None, uuid_var)
    print(proxy.uuid_var)
    proxy.task_start(run_name)


if __name__ == "__main__":
    main()