from typing import List, Tuple

import numpy as np

from alignment.data_structure.eval_data_structure import join_p_withother, Alignment2D
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from bert_api.segmented_instance.segmented_text import get_highlighted_text
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from tlm.qtype.partial_relevance.eval_score_dp_helper import load_eval_result_r
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer


def get_score_for_method(dataset, method, metric):
    run_name = "{}_{}_{}".format(dataset, method, metric)
    eval_res = load_eval_result_r(run_name)
    return eval_res


def main():
    dataset = "dev_word"
    method = "gradient"
    policy_name = "erasure"
    answers: List[Alignment2D] = load_related_eval_answer(dataset, method)
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    eval_res: List[Tuple[str, float]] = get_score_for_method(dataset, method, policy_name)
    score_d = dict(eval_res)
    tokenizer = get_tokenizer()
    target_idx = 1
    for p, a in join_p_withother(problems, answers):
        score = score_d[p.problem_id]
        if score is not None:
            if score > 0.8:
                text1 = p.seg_instance.text1
                print([ids_to_text(tokenizer, text1.get_tokens_for_seg(idx)) for idx in text1.enum_seg_idx()])
                scores = a.contribution.table[target_idx]
                rank = np.argsort(scores)[::-1]
                k = int(0.2 * p.seg_instance.text2.get_seg_len())
                top_k_indices = rank[:k]
                text = get_highlighted_text(tokenizer, top_k_indices, p.seg_instance.text2)
                print(score)
                print(text)


if __name__ == "__main__":
    main()
