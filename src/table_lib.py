import csv
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional, Callable, Any


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
    if file_path.endswith(".csv"):
        f = open(file_path, "r", encoding="utf-8")
        reader = csv.reader(f)
        return reader
    else:
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


@dataclass
class TablePrintHelper:
    column_keys: list[str]
    row_keys: list[str]
    column_key_to_name: Optional[dict[str, str]]
    row_keys_to_name: Optional[dict[str, str]]
    get_value_fn: Callable[[str, str], Any]
    row_head: Optional[str]

    def get_table(self):
        head_val = self.row_head if self.row_head is not None else ""
        head = [head_val]
        if self.column_key_to_name is None:
            columns = self.column_keys
        else:
            columns = [self.column_key_to_name[t] for t in self.column_keys]
        head.extend(columns)

        table = [head]
        for row_key in self.row_keys:
            row = []
            row_key_out = row_key if self.row_keys_to_name is None else self.row_keys_to_name[row_key]
            row.append(row_key_out)
            for column_key in self.column_keys:
                corr_val = self.get_value_fn(row_key, column_key)
                row.append(corr_val)
            table.append(row)
        return table


class DictCache:
    def __init__(self, raw_get_val: Callable):
        self.raw_get_val = raw_get_val
        self.d = {}

    def get_val(self, key):
        if key in self.d:
            return self.d[key]

        v = self.raw_get_val(key)
        self.d[key] = v
        return v
