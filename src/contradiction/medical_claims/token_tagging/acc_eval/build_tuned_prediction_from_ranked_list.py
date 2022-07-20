from typing import List, Dict

from contradiction.medical_claims.token_tagging.acc_eval.label_loaders import load_sent_token_label, SentTokenLabel
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from list_lib import index_by_fn
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def convert_to_binary(rlg: Dict[str, List[TrecRankedListEntry]],
                      threshold):
    for qid, entries in rlg.items():
        for e in entries:
            pass


def build_save(run_name, tag_type, val_label_path):
    rl_path = get_save_path2(run_name, tag_type)
    rlg = load_ranked_list_grouped(rl_path)
    stl: List[SentTokenLabel] = load_sent_token_label(val_label_path)
    stl_d = index_by_fn(lambda s: s.qid, stl)


def main():
    return NotImplemented


if __name__ == "__main__":
    main()