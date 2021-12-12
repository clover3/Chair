import os
from typing import List, Iterable, Callable, Dict, Tuple, Set


from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.passage_scoring import PassageScoringInner
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped
from trec.types import DocID


def run4_rlg_filtered():
    return os.path.join(output_path, "ca_building", "run4", "pc_res.filtered.txt")


class Run4PassageScoring(PassageScoringInner):
    def __init__(self, scorer: FutureScorerI):
        rlg_path = run4_rlg_filtered()
        rlg = load_ranked_list_grouped(rlg_path)
        doc_as_passage_dict: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
            = load_run4_swtt_passage()
        super(Run4PassageScoring, self).__init__(scorer, rlg, doc_as_passage_dict)
