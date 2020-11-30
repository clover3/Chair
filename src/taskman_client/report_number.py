import sys

from taskman_client.task_proxy import get_task_manager_proxy


def main(name, number, condition, field):
    proxy = get_task_manager_proxy()
    proxy.report_number(name, number, condition, field)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])