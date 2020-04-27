import sys

from taskman_client.task_proxy import get_task_manager_proxy


def main(name, number, condition=""):
    proxy = get_task_manager_proxy()
    proxy.report_number(name, number, condition)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])