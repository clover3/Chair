from typing import List, Callable, Dict

from krovetzstemmer import Stemmer

from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from explain.genex.idf_lime import load_idf_fn_for
from list_lib import index_by_fn
from misc_lib import Averager, TEL, NamedAverager, two_digit_float
from tab_print import print_table
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import PartialSegment
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy, IDX_FUNC, \
    IDX_CONTENT
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance, seg_to_text
from tlm.qtype.partial_relevance.eval_metric.doc_modify_fns import get_replace_non_zero, DocReplaceFunc
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import RandomSpan, load_random_span, \
    get_metric_ps_replace
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.mmd_cached_client import get_mmd_cache_client, MMDCacheClient
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer
from trainer.promise import list_future


def run_eval(dataset, method, target_idx, cache_client, wrong_mode=False):
    print("run_eval ENTRY")
    # This implementation removes target query
    # removes dt
    problem_list: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    pid_to_p: Dict[str, RelatedEvalInstance] = index_by_fn(lambda e: e.problem_id, problem_list)
    answers = load_related_eval_answer(dataset, method)
    tokenizer = get_tokenizer()

    preserve_idx = 1 - target_idx
    print("Target_idx: ", target_idx)
    print("preserve_idx: ", preserve_idx)
    print("{} problems {} answers".format(len(problem_list), len(answers)))
    random_spans: List[RandomSpan] = load_random_span()[:100]

    avg_score_per_span = NamedAverager()
    def get_word_pool(e) -> List[List[int]]:
        words = [span.ngram_ids for span in random_spans]
        return words
    get_idf = load_idf_fn_for("tdlt")

    stemmer = Stemmer()
    def get_idf_from_ids(ids):
        s = pretty_tokens(tokenizer.convert_ids_to_tokens(ids))
        tokens = map(stemmer.stem, s.split())
        idf_list = map(get_idf, tokens)
        return sum(idf_list)

    doc_modify_fn: DocReplaceFunc = get_replace_non_zero()
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = cache_client.predict
    join_policy = FuncContentSegJoinPolicy()

    averager = Averager()
    for a in TEL(answers):
        problem = pid_to_p[a.problem_id]
        # print(rei_to_text(tokenizer, problem))
        key = problem.problem_id, target_idx
        target_q_seg = problem.seg_instance.text1.get_tokens_for_seg(target_idx)
        other_q_seg = problem.seg_instance.text1.get_tokens_for_seg(1-target_idx)
        # print("idf of target(removed): ", get_idf_from_ids(target_q_seg))
        # print("idf of other(preserved): ", get_idf_from_ids(other_q_seg))
        word_pool: List[List[int]] = get_word_pool(key)
        if not word_pool:
            raise IndexError()
        # print(a.contribution.table)
        doc_term_scores: List[float] = a.contribution.table[target_idx]
        wrong_term_scores: List[float] = a.contribution.table[1-target_idx]
        if wrong_mode:
            doc_term_scores = wrong_term_scores
        seg_list = test_core_fn(problem, preserve_idx, doc_modify_fn, join_policy, doc_term_scores, word_pool)

        mask_word = tokenizer.convert_tokens_to_ids(["[MASK]"])
        partial_tokens = PartialSegment.init_one_piece(mask_word)
        full_query = problem.seg_instance.text1
        new_query = join_policy.join_tokens(full_query, partial_tokens, preserve_idx)
        new_doc = doc_modify_fn(problem.seg_instance.text2,
                                doc_term_scores, mask_word)
        seg1_text = seg_to_text(tokenizer, new_query)
        seg2_text = seg_to_text(tokenizer, new_doc)
        # print("Replaced query:", seg1_text)
        # print("Replaced doc:", seg2_text)

        eval_score_list = forward_fn(seg_list)
        for idx, s in enumerate(eval_score_list):
            avg_score_per_span.avg_dict[str(idx)].append(s)

        pos_indices: List[int] = [idx for idx, s in enumerate(eval_score_list) if s > 0.5]
        pos_rate = len(pos_indices) / len(random_spans)
        # print(len(pos_indices), "positive")
        pos_spans = [random_spans[i] for i in pos_indices]
        # print(["{0} ({1:.2f})".format(" ".join(s.ngram), s.idf_sum) for s in pos_spans])
        averager.append(pos_rate)
        # for i in pos_indices:
        #     print(str(random_spans[i]))

    print("Average pos rate: ", averager.get_average())
    cache_client.save_cache()
    return averager.get_average(), avg_score_per_span.get_average_dict()


def test_core_fn(problem, preserve_idx, doc_modify_fn, join_policy, doc_term_scores,
                 word_pool) -> List[SegmentedInstance]:
    seg_list: List[SegmentedInstance] = []
    for word in word_pool:
        partial_tokens = PartialSegment.init_one_piece(word)
        full_query = problem.seg_instance.text1
        new_query = join_policy.join_tokens(full_query, partial_tokens, preserve_idx)
        new_doc = doc_modify_fn(problem.seg_instance.text2,
                                doc_term_scores, word)
        new_qd = SegmentedInstance(new_query, new_doc)
        seg_list.append(new_qd)
    return seg_list


def run_eval_w_class(dataset, method, target_idx, cache_client, wrong_mode=False):
    print("run_eval_w_class ENTRY")
    problem_list: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    pid_to_p: Dict[str, RelatedEvalInstance] = index_by_fn(lambda e: e.problem_id, problem_list)
    answers = load_related_eval_answer(dataset, method)
    avg_score_per_span = NamedAverager()
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = cache_client.predict
    eval_metric = get_metric_ps_replace(forward_fn, target_idx, True)
    averager = Averager()
    for a in TEL(answers):
        problem = pid_to_p[a.problem_id]
        wrong_term_scores: List[float] = a.contribution.table[1-target_idx]
        if wrong_mode:
            a.contribution.table[target_idx] = wrong_term_scores
        scores_futures = eval_metric.get_predictions_for_case(problem, a)
        eval_metric.do_duty()
        for i, score in enumerate(list_future(scores_futures)):
            avg_score_per_span.avg_dict[i].append(score)
        pos_rate = eval_metric.convert_future_to_score(scores_futures)
        averager.append(pos_rate)
        eval_metric.reset()
    print("Average pos rate: ", averager.get_average())
    cache_client.save_cache()
    return averager.get_average(), avg_score_per_span.get_average_dict()


def func_replace():
    dataset_name = "dev_sent"
    cache_client: MMDCacheClient = get_mmd_cache_client("localhost")
    table = []
    pos_rate_per_word_d = {}
    for wrong_mode in [False, True]:
        for target in [IDX_FUNC, IDX_CONTENT]:
            key = wrong_mode, target
            pos_rate, pos_rate_per_word = run_eval_w_class(
                dataset_name,
                "exact_match", target, cache_client, wrong_mode)

            pos_rate_per_word_d[key] = pos_rate_per_word
            table.append((key, pos_rate))

    print_table(table)
    random_spans: List[RandomSpan] = load_random_span()[:100]
    columns = list(pos_rate_per_word_d.keys())
    table = []
    for i in range(100):
        span = random_spans[i]
        row = [str(i), span.get_ngram_s(), span.tag, span.idf_sum]
        for c in columns:
            s: float = pos_rate_per_word_d[c][i]
            row.append(two_digit_float(s))
        table.append(row)
    print_table(table)


def main():
    func_replace()


if __name__ == "__main__":
    main()