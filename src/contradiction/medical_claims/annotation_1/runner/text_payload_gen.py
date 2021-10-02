import csv
import os
from typing import List, Tuple

from contradiction.medical_claims.annotation_1.load_data import load_alamri1_all
from cpath import output_path
from misc_lib import exist_or_mkdir


def main():
    data: List[Tuple[int, List[Tuple[str, str]]]] = load_alamri1_all()

    info_lines = []
    payload_lines = []
    for group_no, entries in data:
        for inner_idx, (t1, t2) in enumerate(entries):
            info_e = (group_no, inner_idx, 0)
            info_lines.append(info_e)
            payload_lines.append((t1, t2))

            info_e = (group_no, inner_idx, 1)
            info_lines.append(info_e)
            payload_lines.append((t2, t1))

    assert len(payload_lines) == len(info_lines)

    save_dir = os.path.join(output_path, "alamri_annotation1", "plain_payload")
    exist_or_mkdir(save_dir)

    def save_csv(csv_save_path, rows):
        f_out = csv.writer(open(csv_save_path, "w", newline=""))

        for row in rows:
            str_row = list(map(str, row))
            f_out.writerow(str_row)

    info_save_path = os.path.join(save_dir, "info.csv")
    save_csv(info_save_path, info_lines)

    payload_save_path = os.path.join(save_dir, "payload.csv")
    save_csv(payload_save_path, payload_lines)



if __name__ == "__main__":
    main()