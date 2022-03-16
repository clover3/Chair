import random
from typing import List, Callable

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import Averager
from models.classic.stopword import load_stopwords
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import IDX_FUNC, \
    IDX_CONTENT
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import get_mask_empty_as_span
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_replace_non_target_query, QueryReplaceFunc
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.mmd_cached_client import get_mmd_cache_client, SQLBasedCacheClient


class NotApplicableException(Exception):
    pass


def check_for_content_func(cache_client,
                           forward_fn: Callable[[List[SegmentedInstance]], List[float]],
                           dataset_name,
                           ):
    items = ["FUNC", "CONTENT"]
    for target_idx in [IDX_FUNC, IDX_CONTENT]:
        problems = load_mmde_problem(dataset_name)
        replacee_span_list = get_mask_empty_as_span()
        ref_query_modify_fn: QueryReplaceFunc = get_replace_non_target_query()
        empty_span = replacee_span_list[0].ngram_ids

        def query_modify_fn(query):
            new_query = ref_query_modify_fn(query, target_idx, empty_span)
            return new_query
        print("Keep {} and delete {} ".format(items[target_idx], items[1-target_idx]))
        n_true_after, n_true_before = get_change_rate(forward_fn, problems, query_modify_fn)
        cache_client.save_cache()
        print("{} -> {}".format(n_true_before, n_true_after))


def run_deletions(cache_client,
                  forward_fn: Callable[[List[SegmentedInstance]], List[float]],
                  dataset_name,
                  ):
    problems = load_mmde_problem(dataset_name)
    deletion_rate = Averager()
    stopwords = load_stopwords()
    tokenizer = get_tokenizer()
    def query_modify_fn(query: SegmentedText) -> SegmentedText:
        tokens = tokenizer.convert_ids_to_tokens(query.tokens_ids)
        stopword_indices = []
        for idx, t in enumerate(tokens):
            if t in stopwords:
                stopword_indices.append(idx)

        l = len(query.tokens_ids)
        if not stopword_indices:
            raise NotApplicableException()
        drop_idx = random.sample(stopword_indices, 1)[0]
        new_tokens_ids = query.tokens_ids[:drop_idx] + query.tokens_ids[drop_idx+1:]
        deletion_rate.append(1/l)
        return SegmentedText.from_tokens_ids(new_tokens_ids)
    print("Delete random one stopwords")
    n_true_after, n_true_before = get_change_rate(forward_fn, problems, query_modify_fn)
    cache_client.save_cache()
    print("Average deletion rate:", deletion_rate.get_average())
    print("{} -> {}".format(n_true_before, n_true_after))


def get_change_rate(forward_fn, problems, query_modify_fn):
    def is_true(p):
        return p >= 0.5

    n_true_before = 0
    n_true_after = 0
    for p in problems:
        try:
            query = p.seg_instance.text1
            doc = p.seg_instance.text2
            new_query = query_modify_fn(query)
            seg = SegmentedInstance(query, doc)
            new_seg = SegmentedInstance(new_query, doc)
            before_score, after_score = forward_fn([seg, new_seg])
            if is_true(before_score):
                n_true_before += 1
            if is_true(after_score):
                n_true_after += 1
            # print("{} -> {}".format(query_to_text(query), query_to_text(new_query)))
            # print("{0:.2f} -> {1:.2f}".format(before_score, after_score))
        except NotApplicableException:
            pass
    return n_true_after, n_true_before


def main():
    dataset_name = "dev_sent"
    cache_client: SQLBasedCacheClient = get_mmd_cache_client("localhost")
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = cache_client.predict
    run_deletions(cache_client, forward_fn, dataset_name)


if __name__ == "__main__":
    main()
