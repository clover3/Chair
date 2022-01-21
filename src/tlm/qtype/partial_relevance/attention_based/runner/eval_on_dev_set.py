import os
from typing import Iterator
from typing import List, Tuple

from cache import load_pickle_from, save_to_pickle
from cpath import output_path
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import TimeEstimator
from tlm.data_gen.doc_encode_common import split_window_get_length
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo
from tlm.qtype.enum_util import enum_samples
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import EvalPerQSeg
from tlm.qtype.partial_relevance.attention_based.attention_mask_gradient import AttentionGradientScorer, \
    PredictorAttentionMaskGradient
from tlm.qtype.partial_relevance.attention_based.bert_masking_client import get_localhost_bert_mask_client
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import dist_l2
from tlm.qtype.partial_relevance.attention_based.perturbation_scorer import PerturbationScorer
from tlm.qtype.partial_relevance.eval_data_structure import QDSegmentedInstance
from tlm.qtype.qtype_instance import QTypeInstance


def enum_by_label(qtype_entries: List[QTypeInstance], query_info_dict):
    seen = set()
    n_rel = 0
    n_non_rel = 0
    for e_idx, e in enumerate(qtype_entries):
        if e.label > 0.5:
            n_rel += 1
        else:
            if n_non_rel >= n_rel:
                continue
            n_non_rel += 1
        overlap_key = e.qid
        info: QueryInfo = query_info_dict[e.qid]
        if overlap_key in seen:
            continue
        yield e, info


def enum_qtype_2Y_v_train_120000() -> Iterator[Tuple[QTypeInstance, QueryInfo]]:
    run_name = "qtype_2Y_v_train_120000"
    save_dir = os.path.join(output_path, "qtype", run_name + '_sample')
    _, query_info_dict = load_pickle_from(os.path.join(save_dir, "0"))
    itr = enum_samples(save_dir)
    yield from enum_by_label(itr, query_info_dict)


def get_segmented_inst(tokenizer, e, info) -> QDSegmentedInstance:
    head, tail = info.get_head_tail()
    entity, d_tokens_ids = split_p_h_with_input_ids(e.de_input_ids, e.de_input_ids)
    q_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(info.query))
    q_seg1_indice = [i for i, _ in enumerate(head)]
    base = len(q_seg1_indice) + len(entity)
    q_seg2_indice = [i + base for i, _ in enumerate(tail)]
    q_seg_indices = [q_seg1_indice, q_seg2_indice]
    window_size = 10
    d_seg_len_list = split_window_get_length(d_tokens_ids, window_size)
    n_d_segs = len(d_seg_len_list)
    seg_insts = QDSegmentedInstance(q_tokens_ids, d_tokens_ids.tolist(),
                                    e.label,
                                    n_d_segs, 2,
                                    d_seg_len_list, q_seg_indices)
    return seg_insts


def main():
    max_seq_length = 512
    attention_mask_gradient = PredictorAttentionMaskGradient(2, max_seq_length)
    predictor = get_localhost_bert_mask_client()
    scorer = PerturbationScorer(predictor, max_seq_length, dist_l2)
    scorer2 = AttentionGradientScorer(attention_mask_gradient, max_seq_length)

    contrib_score_list = [scorer, scorer2]
    tokenizer = get_tokenizer()
    eval_model = EvalPerQSeg(predictor, max_seq_length)

    n_predict = 20
    ticker = TimeEstimator(n_predict, "", 9)
    auc_list_list: List[List[Tuple[QDSegmentedInstance, List[float]]]] = [list() for _ in contrib_score_list]
    for idx, (e, info) in enumerate(enum_qtype_2Y_v_train_120000()):
        if idx >= n_predict:
            break
        seg_insts: QDSegmentedInstance = get_segmented_inst(tokenizer, e, info)
        for method_idx, scorer in enumerate(contrib_score_list):
            res = scorer.eval_contribution(seg_insts)
            auc_list = eval_model.eval(seg_insts, res)
            auc_list_list[method_idx].append((seg_insts, auc_list))
        ticker.tick()

    save_to_pickle(auc_list_list, "dev_auc_predict")


if __name__ == "__main__":
    main()
