from typing import List, Iterable, Callable, Dict, Tuple, Set
import sys
from collections import defaultdict, Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np
from transformers import AutoTokenizer
from cpath import output_path
from misc_lib import path_join, TELI

from dataset_specific.msmarco.passage.passage_resource_loader import enum_all_when_corpus, enum_grouped, FourStr
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import build_table
from trainer_v2.per_project.transparency.mmp.when_corpus_based.gradient_computer import GoldPairBasedSampler
import tensorflow as tf

from trec.qrel_parse import load_qrels_structured


def main():
    judgment_path = path_join("data", "msmarco", "qrels.train.tsv")
    qrels = load_qrels_structured(judgment_path)
    itr: Iterable[FourStr] = enum_all_when_corpus()
    for e in itr:
        qid, pid, query, text = e
        if "circa" in text:
            qrel_d = qrels[qid]
            is_rel = pid in qrel_d and qrel_d[pid]
            if is_rel:
                print(f"Circa is in relevant document ({qid}, {pid})")
            else:
                print(f"Circa is in NOT relevant document ({qid}, {pid})")


if __name__ == "__main__":
    main()