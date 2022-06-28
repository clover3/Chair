import os
import sys
import time


def main():
    raw_file_list = sys.argv[1]
    print("wait files:", raw_file_list)
    file_list = raw_file_list.split(",")
    num_files = len(file_list)
    wait = True
    while wait:
        n_done = 0
        n_found = 0
        for file_path in file_list:
            if os.path.exists(file_path):
                n_found += 1
                mtime = os.path.getmtime(file_path)
                now = time.time()
                elapsed = now - mtime
                if elapsed > 30:
                    n_done += 1
        if n_done == num_files:
            print("All ({}) files found".format(n_done))
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
