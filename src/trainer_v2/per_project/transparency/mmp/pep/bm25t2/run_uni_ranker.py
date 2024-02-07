import itertools
import logging
import pickle
import sys
from collections import defaultdict, Counter
from typing import Iterable, Tuple, Callable

from omegaconf import OmegaConf

from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_line_scores_and_eval
from adhoc.other.bm25_retriever_helper import get_bm25_stats_from_conf
from cache import load_pickle_from
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import load_bm25_index_resource_conf
from misc_lib import select_third_fourth, TimeEstimator
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import read_lines
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import get_line_scores_path
from trainer_v2.per_project.transparency.mmp.pep.bm25t2.pep_uni_ranker import PepUniRanker
from misc_lib import path_join


def load_table(conf):
    q_terms = read_lines(conf.q_term_path)
    save_dir = conf.score_save_dir
    table = defaultdict(Counter)
    ticker = TimeEstimator(len(q_terms))
    for idx, q_term in enumerate(q_terms):
        q_term = q_term.strip()
        try:
            for d_term, score_s in tsv_iter(path_join(save_dir, "{}.txt".format(idx))):
                table[q_term][d_term] = float(score_s)
        except FileNotFoundError:
            pass
        ticker.tick()

    return table


def main():
    c_log.info(__file__)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    run_name = conf.run_name
    c_log.setLevel(logging.INFO)
    dataset_conf_path = conf.dataset_conf_path
    dataset_conf = OmegaConf.load(dataset_conf_path)
    dataset_name = dataset_conf.dataset_name
    data_size = dataset_conf.data_size
    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path
    quad_tsv_path = dataset_conf.rerank_payload_path
    scores_path = get_line_scores_path(run_name, dataset_name)
    f = open(scores_path, "w")
    table = load_pickle_from(conf.table_pickle_path)
    # table = load_table(conf)
    # pickle.dump(table, open(conf.table_pickle_path, "wb"))
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf)
    bm25_scoring_fn: Callable[[int, int, int, int], float] = build_bm25_scoring_fn(cdf, avdl)

    ranker = PepUniRanker(table, bm25_scoring_fn, df)
    score_fn = ranker.score

    qd_iter: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    qd_iter = itertools.islice(qd_iter, 1000 * 100)
    ticker = TimeEstimator(data_size)
    for q, d in qd_iter:
        arg = [(q, d)]
        score = score_fn(arg)[0]
        f.write("{}\n".format(score))
        ticker.tick()
    f.close()

    build_ranked_list_from_line_scores_and_eval(
        run_name, dataset_name, judgment_path,
        quad_tsv_path, scores_path,
        metric, do_not_report=True)

    print("QD hit rate : ", ranker.qd_hit.get_suc_prob())
    print("Q Terms match/mis = {}/{}".format(len(ranker.matched_q_terms), len(ranker.missed_q_terms)))


if __name__ == "__main__":
    main()
