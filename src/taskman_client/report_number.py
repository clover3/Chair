import sys

from taskman_client.task_proxy import get_task_manager_proxy


def main(name, number, condition, field):
    proxy = get_task_manager_proxy()
    proxy.report_number(name, number, condition, field)


if __name__ == "__main__":
    name = sys.argv[1]
    number = sys.argv[2]
    condition = sys.argv[3]
    field = sys.argv[4]
    main(name, number, condition, field)
