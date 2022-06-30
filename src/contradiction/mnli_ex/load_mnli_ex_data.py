import csv
import os
from typing import List, NamedTuple

from cpath import data_path


def parse_comma_sep_indices(s):
    tokens = s.split(",")
    tokens = [t for t in tokens if t]
    return list(map(int, tokens))


class MNLIExEntry(NamedTuple):
    data_id: str
    premise: str
    hypothesis: str
    p_indices: List[int]
    h_indices: List[int]

    @classmethod
    def from_dict(cls, d):
        return MNLIExEntry(
            d['data_id'],
            d['premise'],
            d['hypothesis'],
            parse_comma_sep_indices(d['p_indices']),
            parse_comma_sep_indices(d['h_indices']),
        )

prefix_d = {
    "match": "e",
    "mismatch": "n",
    "conflict": "c",
}

mnli_ex_tags = ["match", "mismatch", "conflict"]

def load_mnli_ex(split, label) -> List[MNLIExEntry]:
    tag_prefix = prefix_d[label]
    file_name = "{}_{}.tsv".format(tag_prefix, split)
    file_path = os.path.join(data_path, "nli", "mnli_ex", file_name)
    f = open(file_path, "r")
    reader = csv.reader(f, delimiter='\t', quotechar=None)

    data = []
    for g_idx, row in enumerate(reader):
        if g_idx == 0 :
            columns = row
        else:
            entry = {}
            for idx, column in enumerate(columns):
                entry[column] = row[idx]
            data.append(entry)

    return list(map(MNLIExEntry.from_dict, data))

