import csv
import gzip
import json
from typing import List, Iterable, Callable, Dict, Tuple, Set

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter


def save_tsv(entries, save_path):
    save_f = open(save_path, "w")
    for row in entries:
        out_line = "\t".join(map(str, row))
        save_f.write(out_line + "\n")


def load_tsv(file_path) -> List:
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    table = []
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        table.append(row)
    return table


def load_str_float_tsv(qid, save_path):
    entries = []
    for pid, score in tsv_iter(save_path):
        entries.append((pid, float(score)))
    return qid, entries


class CustomEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        if isinstance(o, float):
            yield format(o, '.4g')
        elif isinstance(o, list):
            yield '['
            first = True
            for value in o:
                if first:
                    first = False
                else:
                    yield ', '
                yield from self.iterencode(value)
            yield ']'
        else:
            yield from super().iterencode(o, _one_shot=_one_shot)


def save_list_to_gz_jsonl(item_list, save_path):
    f_out = gzip.open(save_path, 'wt', encoding='utf8')
    for item in item_list:
        s = json.dumps(item, cls=CustomEncoder)
        f_out.write(s + "\n")
    f_out.close()


def load_list_from_gz_jsonl(save_path, from_json):
    f = gzip.open(save_path, 'rt', encoding='utf8')
    return [from_json(json.loads(line)) for line in f]


def save_number_to_file(save_path, score):
    f = open(save_path, "w")
    f.write(str(score))


def read_term_pair_table(score_path) -> List[Tuple[str, str, float]]:
    itr = tsv_iter(score_path)
    term_gain: List[Tuple[str, str, float]] = []
    for row in itr:
        qt, dt, score = row
        term_gain.append((qt, dt, float(score)))
    return term_gain