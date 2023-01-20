import csv
import os
from typing import List

from contradiction.mnli_ex.nli_ex_common import NLIExEntry, nli_ex_entry_to_sent_token_label
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from cpath import data_path
from data_generator.NLI.enlidef import prefix_d


def load_mnli_ex(split, label) -> List[NLIExEntry]:
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

    return list(map(NLIExEntry.from_dict, data))


def load_mnli_ex_binary_label(split, tag_type) -> List[SentTokenLabel]:
    entries = load_mnli_ex(split, tag_type)
    output: List[SentTokenLabel] = []
    for e in entries:
        output.extend(nli_ex_entry_to_sent_token_label(e, tag_type))
    return output

