import os
import re

from misc_lib import get_dir_dir, get_dir_files


def main():
    root_dir = "/mnt/nfs/work1/allan/youngwookim/models"
    regex = "model_checkpoint_path: \"(.*)\""
    for dir_path in get_dir_dir(root_dir):
        try:
            checkpoint_file = os.path.join(dir_path, "checkpoint")
            f = open(checkpoint_file, "r")
            print(dir_path)
            preserve_name = None
            for line in f:
                if line.startswith("model_checkpoint_path"):
                    print(line)
                    m = re.match(regex, line)
                    found_name = m.group(1)
                    if found_name[0] == "/":
                        preserve_name = found_name
                    else:
                        preserve_name = os.path.join(dir_path, found_name)

            if preserve_name is None:
                print("skip", dir_path)
                continue

            n_file_found = 0
            for file_path in get_dir_files(dir_path):
                if file_path.startswith(preserve_name):
                    n_file_found += 1

            if n_file_found == 0:
                print("preseve target {} not found".format(preserve_name))
                continue

            for file_path in get_dir_files(dir_path):
                if not file_path.startswith(preserve_name) and not file_path.endswith("checkpoint"):
                    print("DETETE : ", file_path)
                    os.remove(file_path)
                else:
                    print("PRESERVE : ", file_path)
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":
    main()

