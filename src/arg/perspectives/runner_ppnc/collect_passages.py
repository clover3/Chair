import json
import os
import sys
from typing import List

from arg.qck.decl import KDP
from misc_lib import get_dir_files
from tab_print import print_table


def collect_unique_passage(dir_path):
    key_set = set()
    unique_passages = []

    def update(j):
        for doc_id, value in j.items():
            j = value
            kdp = KDP(*j['kdp'])
            key = kdp.doc_id, kdp.passage_idx
            if key not in key_set:
                unique_passages.append(kdp)
                key_set.add(key)

    if os.path.isdir(dir_path):
        d = {}
        for file_path in get_dir_files(dir_path):
            if file_path.endswith(".info"):
                j = json.load(open(file_path, "r", encoding="utf-8"))
                update(j)
    else:
        d = json.load(open(dir_path, "r"))
        update(d)
    return unique_passages


def main():
    info_dir = sys.argv[1]
    unique_passages: List[KDP] = collect_unique_passage(info_dir)

    rows = []
    for p in unique_passages:
        row = [p.doc_id, p.passage_idx, " ".join(p.tokens)]
        rows.append(row)

    print_table(rows)


if __name__ == "__main__":
    main()