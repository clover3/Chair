import json
import os

from arg.qck.decl import KDP, qck_convert_map
from arg.qck.prediction_reader import parse_info
from cpath import output_path
from misc_lib import get_dir_files


def load_combine_info_jsons(dir_path, convert_map):
    token_d = {}
    for file_path in get_dir_files(dir_path):
        if file_path.endswith(".info"):
            j = json.load(open(file_path, "r", encoding="utf-8"))
            parse_info(j, convert_map, False)

            for data_id, info in j.items():
                kdp: KDP = info["kdp"]
                key = kdp.doc_id, kdp.passage_idx
                if key in token_d:
                    if str(token_d[key]) != str(kdp.tokens):
                        print(key)
                token_d[key] = kdp.tokens
    # field

def main():
    save_name = "qcknc_val"
    out_dir = os.path.join(output_path, "cppnc")
    info_path = os.path.join(out_dir, save_name + ".info")
    load_combine_info_jsons(info_path, qck_convert_map)


if __name__ == "__main__":
    main()