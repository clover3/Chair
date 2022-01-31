from typing import List

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer, JoinEncoder, pretty_tokens
from tlm.data_gen.doc_encode_common import split_by_window, split_window_get_length
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import softmax_rev_sigmoid, EvalPerQSeg
from tlm.qtype.partial_relevance.attention_based.bert_masking_client import get_localhost_bert_mask_client
from tlm.qtype.partial_relevance.attention_based.perturbation_scorer import PerturbationScorer
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance
from tlm.qtype.partial_relevance.segmented_text import SegmentedText


def cal2(predictor, inst):
    max_seq_length = 512
    join_encoder = JoinEncoder(max_seq_length)
    x0, x1, x2 = join_encoder.join(inst.text1_tokens_ids, inst.text2_tokens_ids)
    new_payload = x0, x1, x2, {}
    output = predictor.predict([new_payload])
    scores = softmax_rev_sigmoid(output)
    return scores


def dist_on_log_space(base, after):
    outputs = np.stack([base, after])
    scores = softmax_rev_sigmoid(outputs)

    base_score = scores[0]
    after_score = scores[1]
    return base_score - after_score


def dist_l2(base, after):
    return np.linalg.norm(np.array(base) - np.array(after))


def main():
    query = 'Benefits of Coffee'
    doc = "Benefits of Coffee Coffee is good for concentration Coffee is delicious Coffee is good for health Coffee is not good"
    query = 'Where is coffee from?'
    doc = "The earliest credible evidence of the drinking of coffee in the form of the modern beverage appears in modern-day Yemen from the middle of the 15th century in Sufi shrines, where coffee seeds were first roasted and brewed in a manner similar to current methods.[2] The Yemenis procured the coffee beans from the Ethiopian Highlands via coastal Somali intermediaries and began cultivation. By the 16th century, the drink had reached the rest of the Middle East and North Africa, later spreading to Europe"
    # doc = "Coffee is good for health "
    tokenizer = get_tokenizer()
    max_seq_length = 512
    q_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))
    d_tokens = tokenizer.tokenize(doc)
    d_tokens_ids = tokenizer.convert_tokens_to_ids(d_tokens)

    assert len(q_tokens_ids) == 5
    q_seg_indices: List[List[int]] = [[0, 1, 3, 4], [2]]
    window_size = 10
    list(split_by_window(d_tokens_ids, window_size))
    d_seg_len_list = split_window_get_length(d_tokens_ids, window_size)
    st = 0
    d_seg_indices = []
    for l in d_seg_len_list:
        ed = st + l
        d_seg_indices.append(list(range(st, ed)))
        st = ed
    text1 = SegmentedText(q_tokens_ids, q_seg_indices)
    text2 = SegmentedText(d_tokens_ids, d_seg_indices)
    inst = SegmentedInstance(text1, text2)
    predictor = get_localhost_bert_mask_client()
    scorer = PerturbationScorer(predictor, max_seq_length, dist_l2)

    # res = scorer.eval_contribution(inst)
    res = scorer.eval_contribution(inst)

    contribution_scores = res.table
    eval_model = EvalPerQSeg(predictor, max_seq_length)
    for q_idx in [0, 1]:
        scores = contribution_scores[q_idx]
        print("Q_idx", q_idx)
        for d_idx, score in enumerate(scores):
            st = d_idx * window_size
            ed = st + window_size
            tokens = d_tokens[st:ed]
            print("{0:.2f} {1}".format(score, pretty_tokens(tokens, True)))
        # print(" ".join(["{0} ({1:.2f})".format(term, score) for term, score in zip(d_tokens, scores)]))

    auc = eval_model.eval(inst, res)
    print('auc', auc)
    eval_model.verbose_print(inst, res)


if __name__ == "__main__":
    main()
