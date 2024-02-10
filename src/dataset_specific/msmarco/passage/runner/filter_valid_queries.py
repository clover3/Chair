import json
import sys

from table_lib import tsv_iter


def main():
    j = json.load(open(sys.argv[1], "r"))
    for qid, query in tsv_iter(sys.argv[2]):
        if qid in j:
            print(f"{qid}\t{query}")


if __name__ == "__main__":
    main()