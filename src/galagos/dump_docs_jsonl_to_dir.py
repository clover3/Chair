import json
import os
import sys
from json import JSONDecodeError

from misc_lib import exist_or_mkdir


def main():
    f = open(sys.argv[1], "r", encoding="utf-8")
    dir_path = sys.argv[2]
    exist_or_mkdir(dir_path)

    for idx, line in enumerate(f):
        try:
            doc = json.loads(line)
            f_out = open(os.path.join(dir_path, doc['id'] + ".html"), "w")
            f_out.write(doc['content'])
            f_out.close()
        except JSONDecodeError:
            print("decode failed at line {}".format(idx))

        ##


if __name__ == "__main__":
    main()

