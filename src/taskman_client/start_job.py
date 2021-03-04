import argparse
import os
import sys

from taskman_client.task_proxy import get_task_proxy

parser = argparse.ArgumentParser(description='')
parser.add_argument("--job_id", default=-1)

def main():
    if 'uuid' in os.environ:
        uuid_var = os.environ['uuid']
    else:
        uuid_var = None
    run_name = sys.argv[1]
    args = parser.parse_args(sys.argv[2:])
    job_id = int(args.job_id)
    proxy = get_task_proxy(None, uuid_var)

    print(proxy.uuid_var)
    if job_id > 0:
        proxy.task_start(run_name, job_id=job_id)
    else:
        proxy.task_start(run_name)

    proxy.task_start(run_name)


if __name__ == "__main__":
    main()