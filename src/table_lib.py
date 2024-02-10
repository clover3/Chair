import csv
from typing import List, Dict, Iterable, Tuple


def read_csv_as_dict(csv_path) -> List[Dict]:
    f = open(csv_path, "r")
    reader = csv.reader(f)
    data = []
    for g_idx, row in enumerate(reader):
        if g_idx == 0:
            columns = row
        else:
            entry = {}
            for idx, column in enumerate(columns):
                entry[column] = row[idx]
            data.append(entry)

    return data


def tsv_iter(file_path) -> Iterable[Tuple]:
    print('tsv_iter', file_path)
    return tsv_iter_gz(file_path)


def tsv_iter_raw(file_path) -> Iterable[Tuple]:
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter='\t')
    return reader


def tsv_iter_gz(file_path) -> Iterable[Tuple]:
    if file_path.endswith(".gz"):
        import gzip
        f = gzip.open(file_path, 'rt', encoding='utf8')
    else:
        f = open(file_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter='\t')
    return reader


def tsv_iter_no_quote(file_path) -> Iterable[Tuple]:
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    for line in f:
        row = line.strip().split("\t")
        yield row


def print_positive_entry(file_read):
    for e in tsv_iter(file_read):
        score = float(e[2])
        if score > 0:
            print("\t".join(e))

