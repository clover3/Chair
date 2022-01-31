from typing import List, Callable

from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance, RelatedEvalAnswer
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import get_100_random_spans, \
    get_ps_replace_w_fixed_word_pool
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.mmd_cached_client import MMDCacheClient, get_mmd_cache_client


def get_score(problem, answer, eval_metric):
    scores_futures = eval_metric.get_predictions_for_case(problem, answer)
    eval_metric.do_duty()
    pos_rate = eval_metric.convert_future_to_score(scores_futures)
    return pos_rate


def get_answer_from_input(target_seg_idx, problem) -> RelatedEvalAnswer:
    s = input()
    indices = [int(t) for t in s.split()]
    return RelatedEvalAnswer.from_indices(indices, target_seg_idx, problem)


def main():
    dataset_name = "dev_sent"
    problems = load_mmde_problem(dataset_name)
    cache_client: MMDCacheClient = get_mmd_cache_client("localhost")
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = cache_client.predict
    target_idx = 0
    replacee_span_list = get_100_random_spans()
    replace_precision = get_ps_replace_w_fixed_word_pool(forward_fn, target_idx, replacee_span_list, "precision")
    replace_recall = get_ps_replace_w_fixed_word_pool(forward_fn, target_idx, replacee_span_list, "recall")

    target_pids = ["1000000-D2056829-29", "1000000-D2056829-30",
                   "1006987-D215775-9", "1006987-D215775-22"]
    tokenizer = get_tokenizer()
    problems = [p for p in problems if p.problem_id in target_pids]
    print("Target_idx=", target_idx)
    for p in problems:
        stop = False
        print("Query: ", p.seg_instance.text1.get_readable_rep(tokenizer))
        print("Document: ", p.seg_instance.text2.get_readable_rep(tokenizer))
        try:
            while not stop:
                answer = get_answer_from_input(target_idx, p)
                s1 = get_score(p, answer, replace_precision)
                s2 = get_score(p, answer, replace_recall)
                print(s1, s2)
        except ValueError:
            pass


if __name__ == "__main__":
    main()
