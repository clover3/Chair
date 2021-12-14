import csv
import os
from typing import Dict, Tuple, List

from arg.counter_arg_retrieval.build_dataset.passage_scoring.passage_scoring import PassageScoringInner
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.split_documents import load_ca3_swtt_passage
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped
from trec.types import DocID


class Run3PassageScoring(PassageScoringInner):
    def __init__(self, scorer: FutureScorerI):
        rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.filtered.dummy.txt")
        rlg = load_ranked_list_grouped(rlg_path)
        doc_as_passage_dict: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
            = load_ca3_swtt_passage()
        super(Run3PassageScoring, self).__init__(scorer, rlg, doc_as_passage_dict)


def load_tsv_query(file_path):
    data = []
    for row in csv.reader(open(file_path, "r"), delimiter="\t"):
        data.append((row[0], row[1]))
    return data


def load_premise_queries():
    query_path = os.path.join(output_path, "ca_building", "run3", "queries", "premise_query.tsv")
    return load_tsv_query(query_path)


def load_manual_queries():
    query_path = os.path.join(output_path, "ca_building", "run3", "queries", "manual_query.tsv")
    return load_tsv_query(query_path)