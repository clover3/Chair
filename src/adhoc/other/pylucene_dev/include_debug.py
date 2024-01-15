print("1")
import logging
print("2")
import sys
print("3")
from omegaconf import OmegaConf
print("4")
from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
print("5")
print("6")
from trainer_v2.per_project.transparency.mmp.table_readers import load_align_scores
print("7")
from trainer_v2.per_project.transparency.mmp.lucene_based.pylucene_backed_bm25 import get_pylucene_back_bm25_retriever
print("8")
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
print("9")


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    value_mapping: Dict[str, Dict[str, float]] = load_align_scores(conf.table_path)
    bm25_conf = OmegaConf.load(conf.bm25conf_path)


if __name__ == "__main__":
    main()