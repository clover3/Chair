import logging
import sys
from omegaconf import OmegaConf
from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.table_readers import load_align_scores
from trainer_v2.per_project.transparency.mmp.lucene_based.pylucene_backed_bm25 import get_pylucene_back_bm25_retriever
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator


def main():
    c_log.setLevel(logging.DEBUG)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    value_mapping: Dict[str, Dict[str, float]] = load_align_scores(conf.table_path)
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    retriever = get_pylucene_back_bm25_retriever(bm25_conf, value_mapping)
    run_retrieval_eval_report_w_conf(conf, retriever, False)


if __name__ == "__main__":
    main()
