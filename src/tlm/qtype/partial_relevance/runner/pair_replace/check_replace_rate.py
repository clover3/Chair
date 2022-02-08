from typing import List, Callable, Dict, Tuple

from krovetzstemmer import Stemmer

from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from explain.genex.idf_lime import load_idf_fn_for
from list_lib import index_by_fn, right
from misc_lib import NamedAverager, average
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import IDX_FUNC, \
    IDX_CONTENT
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import get_ps_replace_w_fixed_word_pool, \
    get_100_random_spans, ReplaceeSpan, get_ps_replace_inner
from tlm.qtype.partial_relevance.eval_score_dp_helper import save_eval_result_r
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.mmd_cached_client import get_mmd_cache_client, SQLBasedCacheClient
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer
from trainer.promise import list_future


def run_eval_w_class(dataset,
                     method,
                     target_idx,
                     eval_metric,
                     wrong_mode=False) -> List[Tuple[str, float]]:
    print("run_eval_w_class ENTRY")
    problem_list: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    pid_to_p: Dict[str, RelatedEvalInstance] = index_by_fn(lambda e: e.problem_id, problem_list)
    answers = load_related_eval_answer(dataset, method)
    avg_score_per_span = NamedAverager()
    score_per_answer: List[Tuple[str, float]] = []
    for a in answers:
        problem = pid_to_p[a.problem_id]
        wrong_term_scores: List[float] = a.contribution.table[1-target_idx]
        if wrong_mode:
            a.contribution.table[target_idx] = wrong_term_scores
        scores_futures = eval_metric.get_predictions_for_case(problem, a)
        eval_metric.do_duty()
        for i, score in enumerate(list_future(scores_futures)):
            avg_score_per_span.avg_dict[i].append(score)
        pos_rate = eval_metric.convert_future_to_score(scores_futures)
        score_per_answer.append((a.problem_id, pos_rate))
        eval_metric.reset()
    print("Average pos rate: ", average(right(score_per_answer)))
    return score_per_answer


def build_get_word_pool_by_neighbor_idf(
        get_idf,
        problems: List[RelatedEvalInstance],
        target_seg_idx,
        replacee_span_list: List[ReplaceeSpan]) -> Callable[[str], List[List[int]]]:

    tokenizer = get_tokenizer()
    spans_per_problems = {}
    for p in problems:
        tokens_to_be_replaced = p.seg_instance.text1.get_tokens_for_seg(1-target_seg_idx)
        s = ids_to_text(tokenizer, tokens_to_be_replaced)
        words = s.split()
        idf_sum = sum([get_idf(w) for w in words])
        def dist(span: ReplaceeSpan):
            return abs(span.idf_sum - idf_sum)

        replacee_span_list.sort(key=dist)
        selected_spans = replacee_span_list[:10]
        spans_per_problems[p.problem_id] = [s.ngram_ids for s in selected_spans]

    def get_word_pool(problem_id):
        return spans_per_problems[problem_id]

    return get_word_pool



def run_all_combinations():
    dataset_name = "dev_sent"
    # method = "exact_match"
    method = "exact_match_noise0.1"
    cache_client: SQLBasedCacheClient = get_mmd_cache_client("localhost")
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = cache_client.predict
    # word_set = "emptyword"
    # replacee_span_list = get_mask_empty_as_span()
    word_set = "100words"
    replacee_span_list = get_100_random_spans()
    run_for_case(cache_client, dataset_name, forward_fn, method, replacee_span_list, word_set)


def run_for_case(cache_client, dataset_name, forward_fn, method, replacee_span_list, word_set):
    wrong_mode = False

    scores_d = {}
    for option_as_metric in ["recall", "precision"]:
        for target_idx in [IDX_FUNC, IDX_CONTENT]:
            key = option_as_metric, wrong_mode, target_idx
            eval_metric = get_ps_replace_w_fixed_word_pool(forward_fn, target_idx, replacee_span_list, option_as_metric)
            score_per_answer: List[Tuple[str, float]] = run_eval_w_class(
                dataset_name, method, target_idx, eval_metric, wrong_mode)
            cache_client.save_cache()
            policy_name = f"replace_{option_as_metric}_{word_set}_{target_idx}"
            run_name = f"{dataset_name}_{method}_{policy_name}"
            scores_d[key] = score_per_answer
            save_eval_result_r(score_per_answer, run_name)
            # pos_rate_per_word_d[key] = pos_rate_per_word


def run_all_combinations2():
    dataset_name = "dev_sent"
    # method = "exact_match"
    method = "exact_match_noise0.1"
    cache_client: SQLBasedCacheClient = get_mmd_cache_client("localhost")
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = cache_client.predict
    word_set = "close10"
    replacee_span_list = get_100_random_spans()
    run_for_case2(cache_client, dataset_name, forward_fn, method, replacee_span_list, word_set)


def run_for_case2(cache_client, dataset_name, forward_fn, method, replacee_span_list, word_set):
    get_idf_inner = load_idf_fn_for("tdlt")
    stemmer = Stemmer()
    def get_idf(word):
        return get_idf_inner(stemmer.stem(word))

    wrong_mode = False
    scores_d = {}
    for option_as_metric in ["recall", "precision"]:
        for target_idx in [IDX_FUNC, IDX_CONTENT]:
            problems = load_mmde_problem(dataset_name)
            get_word_pool = build_get_word_pool_by_neighbor_idf(get_idf, problems, target_idx, replacee_span_list)
            key = option_as_metric, wrong_mode, target_idx
            eval_metric = get_ps_replace_inner(forward_fn, target_idx, option_as_metric, get_word_pool)
            score_per_answer: List[Tuple[str, float]] = run_eval_w_class(
                dataset_name, method, target_idx, eval_metric, wrong_mode)
            cache_client.save_cache()
            policy_name = f"replace_{option_as_metric}_{word_set}_{target_idx}"
            run_name = f"{dataset_name}_{method}_{policy_name}"
            scores_d[key] = score_per_answer
            save_eval_result_r(score_per_answer, run_name)
            # pos_rate_per_word_d[key] = pos_rate_per_word


def main():
    run_all_combinations2()


if __name__ == "__main__":
    main()