import os
import subprocess
import sys


# [Python Daemon]
# [Shell script executing python]
from utils.wait_file import wait_file


def execute_command(cmd):
    print('execute_command:', cmd)
    proc = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    s = proc.stdout + proc.stderr
    print(s)


def main():
    file_to_wait = sys.argv[1]
    cmd_to_be_executed = sys.argv[2]

    if wait_file(file_to_wait):
        print("Wait done")
        execute_command(cmd_to_be_executed)
    else:
        print("wait file failed")


if __name__ == "__main__":
    main()
