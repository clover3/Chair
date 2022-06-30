import os
from typing import List

from contradiction.mnli_ex.load_mnli_ex_data import MNLIExEntry, load_mnli_ex, mnli_ex_tags
from cpath import data_path
from list_lib import lflatten
from trec.trec_parse import write_trec_relevance_judgement
from trec.types import TrecRelevanceJudgementEntry


def convert_mnli_ex_to_trec_style(items: List[MNLIExEntry]) -> List[TrecRelevanceJudgementEntry]:
    return lflatten(map(convert_mnli_ex_entry_to_trec_entries, items))


def convert_mnli_ex_entry_to_trec_entries(e: MNLIExEntry) -> List[TrecRelevanceJudgementEntry]:
    output = []
    todo = [
        ("prem", e.p_indices),
        ("hypo", e.h_indices),
    ]
    for sent_type, indices in todo:
        query_id = "{}_{}".format(e.data_id, sent_type)
        for idx in indices:
            doc_id = str(idx)
            judge = TrecRelevanceJudgementEntry(query_id, doc_id, 1)
            output.append(judge)
    return output


def do_convert_save_trec_style(split, label):
    entries = load_mnli_ex(split, label)
    rel_entries = convert_mnli_ex_to_trec_style(entries)
    save_path = os.path.join(data_path, "nli", "mnli_ex", "trec_style", "{}_{}.txt".format(label, split))
    write_trec_relevance_judgement(rel_entries, save_path)


def main():
    for split in ["dev", "test"]:
        for label in mnli_ex_tags:
            do_convert_save_trec_style(split, label)


if __name__ == "__main__":
    main()