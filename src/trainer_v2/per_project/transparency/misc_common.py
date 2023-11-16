import csv
import gzip
import json
from typing import List, Iterable, Callable, Dict, Tuple, Set, Union


def save_tsv(entries, save_path):
    save_f = open(save_path, "w")
    for row in entries:
        out_line = "\t".join(map(str, row))
        save_f.write(out_line + "\n")


def load_tsv(file_path) -> List:
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    table = []
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        table.append(row)
    return table


def load_str_float_tsv(qid, save_path):
    entries = []
    for pid, score in load_tsv(save_path):
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


def read_term_pair_table_w_score(score_path) -> List[Tuple[str, str, float]]:
    itr = load_tsv(score_path)
    output: List[Tuple[str, str, float]] = []
    for row in itr:
        qt, dt, score = row
        output.append((qt, dt, float(score)))
    return output


def read_term_pair_table(score_path) -> List[Tuple[str, str]]:
    itr = load_tsv(score_path)
    term_pair: List[Tuple[str, str]] = []
    for row in itr:
        if len(row) == 3:
            qt, dt, _score = row
        elif len(row) == 2:
            qt, dt = row
        else:
            raise ValueError("Expect each row to have 2 or 3 columns but it has {} columns"
                             .format(len(row)))
        term_pair.append((qt, dt))
    return term_pair



def save_term_pair_scores(
        score_itr: Union[
            Iterable[Tuple[str, str, float]],
            Iterable[Tuple[Tuple[str, str], float]]
        ],
        save_path):
    out_f = open(save_path, "w")

    for row in score_itr:
        if len(row) == 2:
            (q_term, d_term), score = row
        elif len(row) == 3:
            q_term, d_term, score = row
        else:
            raise ValueError("Expect each row to have 2 or 3 columns but it has {} columns"
                             .format(len(row)))

        out_f.write(f"{q_term}\t{d_term}\t{score}\n")
