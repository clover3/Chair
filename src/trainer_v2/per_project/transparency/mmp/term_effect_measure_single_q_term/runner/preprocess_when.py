import pickle
import sys
from typing import List, Iterable

from dataset_specific.msmarco.passage.passage_resource_loader import enum_grouped, FourItem
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_measure_single_q_term.path_helper import get_precompute_ranked_list_save_path
from trainer_v2.per_project.transparency.mmp.term_effect_measure_single_q_term.precompute_ranked_list import precompute_ranked_list
from trec.qrel_parse import load_qrels_structured


def main():
    idx = sys.argv[1]
    itr = tsv_iter(path_join(output_path, "msmarco", "passage", "when_full_re", str(idx)))
    corpus_name = f"when_full_re_{idx}"
    judgment_path = path_join("data", "msmarco", "qrels.train.tsv")
    qrels = load_qrels_structured(judgment_path)
    g_itr: Iterable[List[FourItem]] = enum_grouped(itr)
    bm25 = get_bm25_mmp_25_01_01()
    e_ranked_list_list, doc_id_to_tf, qid_doc_id_to_target_tf = precompute_ranked_list(g_itr, bm25, "when", qrels)
    items = {
        'erll': e_ranked_list_list,
        'doc_id_to_tf': doc_id_to_tf,
        'qid_doc_id_to_target_tf': qid_doc_id_to_target_tf,
    }

    for key, obj in items.items():
        save_path = get_precompute_ranked_list_save_path(corpus_name, key)
        pickle.dump(obj, open(save_path, "wb"))


if __name__ == "__main__":
    main()