from typing import List

from contradiction.mnli_ex.load_mnli_ex_data import load_mnli_ex
from contradiction.mnli_ex.nli_ex_common import NLIExEntry, get_nli_ex_entry_qid
from data_generator.NLI.enlidef import mnli_ex_tags, is_mnli_ex_target
from contradiction.mnli_ex.path_helper import get_mnli_ex_trec_style_label_path
from trec.trec_parse import write_trec_relevance_judgement
from trec.types import TrecRelevanceJudgementEntry


def convert_mnli_ex_entry_to_trec_entries(e: NLIExEntry, tag_type) -> List[TrecRelevanceJudgementEntry]:
    output = []
    todo = [("prem", e.p_indices), ("hypo", e.h_indices)]
    seen = set()

    for sent_type, indices in todo:
        if is_mnli_ex_target(tag_type, sent_type):
            query_id = get_nli_ex_entry_qid(e, sent_type)
            for idx in indices:
                doc_id = str(idx)
                if (query_id, doc_id) in seen:
                    print((query_id, doc_id), "is seen")
                    pass
                else:
                    judge = TrecRelevanceJudgementEntry(query_id, doc_id, 1)
                    output.append(judge)
                    seen.add((query_id, doc_id))
    return output


def do_convert_save_trec_style(split, label):
    entries = load_mnli_ex(split, label)
    rel_entries = []
    for e in entries:
        rel_entries.extend(convert_mnli_ex_entry_to_trec_entries(e, label))
    save_path = get_mnli_ex_trec_style_label_path(label, split)
    write_trec_relevance_judgement(rel_entries, save_path)


def main():
    for split in ["dev", "test"]:
        for label in mnli_ex_tags:
            do_convert_save_trec_style(split, label)


if __name__ == "__main__":
    main()