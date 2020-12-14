
import json
import sys


def drop_tokens(file_path, out_file_path):
    j = json.load(open(file_path, "r", encoding="utf-8"))
    for data_id, info in j.items():
        info['tokens'] = []
        info['seg_ids'] = []
    json.dump(j, open(out_file_path, "w"))


if __name__ == "__main__":
    drop_tokens(sys.argv[1], sys.argv[2])
