import functools
from typing import List, Tuple

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text, JoinEncoder
from misc_lib import two_digit_float, get_second
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import get_drop_indices, \
    indices_to_mask_dict
from tlm.qtype.partial_relevance.attention_based.attention_mask_gradient import PredictorAttentionMaskGradient, \
    AttentionGradientScorer
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import dist_l2
from tlm.qtype.partial_relevance.bert_mask_interface.bert_masking_client import get_localhost_bert_mask_client
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, rei_to_text, \
    ContributionSummary
from tlm.qtype.partial_relevance.loader import load_mmde_problem


def mmd_item():
    problems: List[RelatedEvalInstance] = load_mmde_problem("dev_sent")
    tokenizer = get_tokenizer()
    for p in problems:
        print(rei_to_text(tokenizer, p))
        yield p.seg_instance


def logits_to_str(logits):
    logits_str = ", ".join(map(two_digit_float, logits))
    return logits_str


# when % of preserve changes, how much does logits/probs change?
def main():
    max_seq_length = 512
    attention_mask_gradient = PredictorAttentionMaskGradient(2, max_seq_length)
    predictor = get_localhost_bert_mask_client()
    scorer = AttentionGradientScorer(attention_mask_gradient, max_seq_length)
    join_encoder = JoinEncoder(max_seq_length)
    num_step = 10
    keep_portion_list = [i / num_step for i in range(num_step)]
    tokenizer = get_tokenizer()
    ids_to_text_fn = functools.partial(ids_to_text, tokenizer)
    for grouped_inst in mmd_item():
        text1_flat = SegmentedText.from_tokens_ids(grouped_inst.text1.tokens_ids)
        text2_flat = SegmentedText.from_tokens_ids(grouped_inst.text2.tokens_ids)
        inst = SegmentedInstance(text1_flat, text2_flat)
        cs: ContributionSummary = scorer.eval_contribution(inst)
        x0, x1, x2 = join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
        indice_score_list = []
        for q_idx, d_idx in inst.enum_seg_indice_pairs():
            k = q_idx, d_idx
            v = cs.table[q_idx][d_idx]
            indice_score_list.append((k, v))
        print("{} pairs".format(len(indice_score_list)))
        indice_score_list.sort(key=get_second, reverse=True)
        get_drop_indices_fn = functools.partial(get_drop_indices, indice_score_list)
        drop_indices_per_case: List[List[Tuple[int, int]]] = list(map(get_drop_indices_fn, keep_portion_list))
        mask_d_itr = list(map(indices_to_mask_dict, drop_indices_per_case))
        mask_d_itr_conv = list(map(inst.translate_mask_d, mask_d_itr))
        payload = []
        for mask_d in mask_d_itr_conv:
            payload_item = x0, x1, x2, mask_d
            payload.append(payload_item)

        base_item = x0, x1, x2, {}
        payload_ex = [base_item] + payload
        output = predictor.predict(payload_ex)
        drop_logits_list = output[1:]
        base_logits = output[0]
        print("Base logits\t({})".format(logits_to_str(base_logits)))
        for j in range(len(keep_portion_list)):
            logits = drop_logits_list[j]
            keep_portion = keep_portion_list[j]
            drop_indices: List[Tuple[int, int]] = drop_indices_per_case[j]
            kept_indices = []
            for k, _ in indice_score_list:
                if k in drop_indices:
                    pass
                else:
                    kept_indices.append(k)
            n_drop = len(drop_indices)
            n_kept = len(kept_indices)
            kept_indices.sort()
            def convert_indices_pair(k_pair):
                k1, k2 = k_pair
                c1 = ids_to_text_fn(inst.text1.get_tokens_for_seg(k1))
                c2 = ids_to_text_fn(inst.text2.get_tokens_for_seg(k2))
                return "({}, {})".format(c1, c2)
            text_list = map(convert_indices_pair, kept_indices)
            kept_tokens_str = " ".join(text_list)
            kept_tokens_str = ""
            error = dist_l2(base_logits, logits)
            logits_str = logits_to_str(logits)
            print(f"{keep_portion}\t{n_kept}\t{n_drop}\t({logits_str})\t{error:.2f}\t{kept_tokens_str}")


if __name__ == "__main__":
    main()