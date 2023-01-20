import csv
import os
from typing import List, NamedTuple, Iterator

from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from cpath import data_path
from data_generator.NLI.enlidef import prefix_d, is_mnli_ex_target


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


def mnli_ex_entry_to_sent_token_label(e: MNLIExEntry, tag_type) -> Iterator[SentTokenLabel]:
    todo = [
        ("prem", e.p_indices, e.premise),
        ("hypo", e.h_indices, e.hypothesis)
    ]
    for sent_type, indices, text in todo:
        if is_mnli_ex_target(tag_type, sent_type):
            n_tokens = len(text.split())
            binary = [1 if i in indices else 0 for i in range(n_tokens)]
            yield SentTokenLabel(
                get_mnli_ex_entry_qid(e, sent_type),
                binary
            )


def get_mnli_ex_entry_qid(e, sent_type):
    query_id = "{}_{}".format(e.data_id, sent_type)
    return query_id


def load_mnli_ex_binary_label(split, tag_type) -> List[SentTokenLabel]:
    entries = load_mnli_ex(split, tag_type)
    output: List[SentTokenLabel] = []
    for e in entries:
        output.extend(mnli_ex_entry_to_sent_token_label(e, tag_type))
    return output

