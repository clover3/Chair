import pickle
from typing import List

from cache import load_from_pickle
from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter
from list_lib import left
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_measure.compute_gains import compute_alignment_gains, ScoringModel, \
    EffectiveRankedListInfo
from cpath import output_path
from misc_lib import path_join, get_second
import numpy as np

from trainer_v2.per_project.transparency.mmp.term_effect_measure.path_helper import get_precompute_ranked_list_save_path


def compute_gains_with(corpus_name, term_targets: List[str]) -> np.array:
    """
    :param corpus_name:
    :param term_targets:
    :return: 2D array of [len(qid), len(term_target)]
    """
    target_q_term = "when"
    c_log.info("Loading bm25")
    bm25 = get_bm25_mmp_25_01_01()

    qtw = bm25.term_idf_factor(target_q_term)
    sm = ScoringModel(bm25.core.k1, bm25.core.b, bm25.core.avdl, qtw)
    doc_id_to_tf, e_ranked_list_list, qid_doc_id_to_target_tf = load_resources(corpus_name)
    c_log.info("Computing")
    gain_arr = compute_alignment_gains(
        e_ranked_list_list, doc_id_to_tf, qid_doc_id_to_target_tf,
        term_targets, sm.get_score)
    c_log.info("Done")
    return gain_arr


def load_resources(corpus_name):
    keys = ['erll', 'doc_id_to_tf', 'qid_doc_id_to_target_tf']
    items = {}
    for key in keys:
        save_path = get_precompute_ranked_list_save_path(corpus_name, key)
        items[key] = pickle.load(open(save_path, "rb"))
    return items['doc_id_to_tf'], items['erll'], items['qid_doc_id_to_target_tf']


def main():
    source_path = path_join(output_path, "msmarco", "passage", "when_trained_saved", "dev_167")
    terms: List[str] = left(tsv_iter(source_path))
    idx = 0

    corpus_name = f"when_{idx}"
    gain_arr = compute_gains_with(corpus_name, terms)

    gain_summary = np.sum(gain_arr, axis=0)
    paired = list(zip(terms, gain_summary))
    paired.sort(key=get_second, reverse=True)
    for t, s in paired:
        print(t, s)



if __name__ == "__main__":
    main()