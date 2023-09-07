import csv
from typing import List, Dict, Iterable, Tuple


def read_csv_as_dict(csv_path) -> List[Dict]:
    f = open(csv_path, "r")
    reader = csv.reader(f)
    data = []
    for g_idx, row in enumerate(reader):
        if g_idx == 0 :
            columns = row
        else:
            entry = {}
            for idx, column in enumerate(columns):
                entry[column] = row[idx]
            data.append(entry)

    return data


def tsv_iter(file_path) -> Iterable[Tuple]:
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter='\t')
    return reader