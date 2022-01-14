import csv
from typing import Dict

from arg.perspectives.load import get_perspective_dict
from table_lib import read_csv_as_dict


def main():
    p = "C:\\work\\Code\\Chair\\output\\ca_building\\run5\\CA - Run5 Queryset - Claim and Perspectives.csv"
    table = read_csv_as_dict(p)
    save_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run5\\query_set.csv"

    p_dict: Dict[int, str] = get_perspective_dict()
    p_dict_rev = {v: k for k, v in p_dict.items()}
    new_rows = []
    for row in table:
        p_text = row['Perspective']
        pid = p_dict_rev[p_text]
        cid = row['CID']
        qid = "{}_{}".format(cid, pid)
        row['pid'] = pid
        row['qid'] = qid
        new_rows.append(row)

    keys = list(new_rows[0].keys())
    out_columns = [k.lower() for k in keys]
    writer = csv.writer(open(save_path, "w",  newline='', encoding="utf-8"))
    writer.writerow(out_columns)
    for row in new_rows:
        out_row = []
        for k in keys:
            out_row.append(row[k])
        writer.writerow(out_row)


if __name__ == "__main__":
    main()