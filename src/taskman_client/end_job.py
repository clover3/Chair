import os
import sys

from taskman_client.task_proxy import get_task_proxy


def main():
    if 'uuid' in os.environ:
        uuid_var = os.environ['uuid']
    elif len(sys.argv) >= 3:
        uuid_var = sys.argv[2]
    else:
        uuid_var = None

    print("UUID:", uuid_var)
    ## end
    run_name = sys.argv[1]
    proxy = get_task_proxy(None, uuid_var)
    proxy.task_complete(run_name, "")


if __name__ == "__main__":
    main()