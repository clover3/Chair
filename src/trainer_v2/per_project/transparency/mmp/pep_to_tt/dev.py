import logging
import sys

import tensorflow as tf
from omegaconf import OmegaConf

from adhoc.bm25_class import BM25Bare
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf
from dataset_specific.msmarco.passage.path_helper import get_train_triples_partition_path
from list_lib import apply_batch
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TextRep
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_common import build_term_pair_scorer, get_pep_predictor


def main_train_loop(conf):
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    get_pep_top_k = get_pep_predictor(conf)

    def dummy_term_weight_predictor(q_term, d_term):
        return tf.constant(0.1)

    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf, None)
    bm25 = BM25Bare(df, len(dl), avdl, b=1.4)
    def score_document(q: TextRep, d: TextRep, relation: str):
        output_score = 0
        for q_term, qtf, _ in q.get_bow():
            exact_match_cnt: int = d.counter[q_term]
            top_k_terms: list[str] = get_pep_top_k(q_term, d.counter.keys())

            if exact_match_cnt:
                score_per_q_term: float = bm25.per_term_score(
                    q_term, qtf, exact_match_cnt, d.get_sp_size())
            elif top_k_terms:
                top_term = top_k_terms[0]
                # We use top score only
                term_freq_tensor: tf.Tensor = dummy_term_weight_predictor(q_term, top_term)
                score_per_q_term: tf.Tensor = bm25.per_term_score(
                    q_term, qtf, term_freq_tensor, d.get_sp_size())

                d_term_str = ", ".join(top_k_terms[:10])
                print(f"{q_term}: {d_term_str}")
                # for i, term in enumerate(top_k_terms[:3]):
                #     c_log.info("%s (%s, %d %s)", relation, q_term, i, term)
            else:
                score_per_q_term = 0

            output_score += score_per_q_term
        return output_score

    model, term_weight_predictor = build_term_pair_scorer(conf.init_checkpoint)
    optimizer = AdamWeightDecay(learning_rate=1e-5,
                                exclude_from_weight_decay=[],
                                )

    part_no = 0
    raw_train_iter = tsv_iter(get_train_triples_partition_path(part_no))
    tokenizer = get_tokenizer()
    batch_size = 16
    for batch_items in apply_batch(raw_train_iter, batch_size):
        loss_arr = []
        for (q, d_pos, d_neg) in batch_items:
            print(f"query: {q}")
            print(f"d_pos: {d_pos}")
            print(f"d_neg: {d_neg}")

            q = TextRep.from_text(tokenizer, q)
            d_pos: TextRep = TextRep.from_text(tokenizer, d_pos)
            d_neg: TextRep = TextRep.from_text(tokenizer, d_neg)

            s_pos = score_document(q, d_pos, "pos")
            s_neg = score_document(q, d_neg, "neg")
            diff = s_pos - s_neg
            hinge_loss = tf.maximum(1.0 - diff, 0)
            loss = hinge_loss
            loss_arr.append(loss)
        loss_avg = tf.reduce_mean(loss_arr)
        loss_np = loss_avg.numpy()
        c_log.info("Loss=%f", float(loss_np))


def main():
    c_log.setLevel(logging.INFO)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    main_train_loop(conf)


if __name__ == "__main__":
    main()
