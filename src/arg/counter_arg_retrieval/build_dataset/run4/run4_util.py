import os
from typing import List, Dict, Tuple

from arg.counter_arg_retrieval.build_dataset.passage_scoring.passage_scoring import PassageScoringInner
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import FutureScorerI
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cache import load_from_pickle
from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped
from trec.types import DocID


def run4_rlg_filtered():
    return os.path.join(output_path, "ca_building", "run4", "pc_res.filtered2.txt")


def load_run4_swtt_passage() -> Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]]:
    return load_from_pickle("ca_run4_swtt_passages")


def show_run4_swtt_count():
    d = load_run4_swtt_passage()

    n_doc = len(d)

    def get_n_passage(doc):
        doc_segs, passage_ranges = doc
        return len(passage_ranges)

    n_passage_all = sum(map(get_n_passage, d.values()))

    print(f"{n_doc} docs, {n_passage_all} passage, {n_passage_all/n_doc} avg")


class Run4PassageScoring(PassageScoringInner):
    def __init__(self, scorer: FutureScorerI):
        rlg_path = run4_rlg_filtered()
        rlg = load_ranked_list_grouped(rlg_path)
        doc_as_passage_dict: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
            = load_run4_swtt_passage()
        super(Run4PassageScoring, self).__init__(scorer, rlg, doc_as_passage_dict)


def main():
    show_run4_swtt_count()


if __name__ == "__main__":
    main()