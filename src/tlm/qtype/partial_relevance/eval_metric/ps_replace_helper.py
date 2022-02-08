from typing import NamedTuple, List

from cache import load_pickle_from
from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.eval_metric.ps_replace import PSReplace, discretized_average, discretized_average_minus
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_not_related_pair_replace_fn, \
    get_related_pair_replace_fn
from tlm.qtype.partial_relevance.mmd_cached_client import get_mmd_cache_client


class ReplaceeSpan(NamedTuple):
    ngram: List[str]
    tag: str
    idf_sum: float
    ngram_ids: List[int]

    def __str__(self):
        return "RandomSpan({0}, {1}, {2:.2f})".format(" ".join(self.ngram), self.tag, self.idf_sum)

    def get_ngram_s(self):
        return " ".join(self.ngram)


def load_random_span() -> List[ReplaceeSpan]:
    save_path = at_output_dir("qtype", "msmarco_random_spans.pickle")
    data = load_pickle_from(save_path)
    tokenizer = get_tokenizer()

    def parse_entry(e) -> ReplaceeSpan:
        ngram_s, tag, idf_sum = e
        ngram_ids: List[int] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join(ngram_s)))
        return ReplaceeSpan(ngram_s, tag, idf_sum, ngram_ids)

    outputs: List[ReplaceeSpan] = []
    for list_per_doc in data:
        outputs.extend(map(parse_entry, list_per_doc))
    return outputs


def get_100_random_spans() -> List[ReplaceeSpan]:
    random_spans: List[ReplaceeSpan] = load_random_span()[:100]
    return random_spans


def get_mask_empty_as_span() -> List[ReplaceeSpan]:
    span = ReplaceeSpan([], 'None', 1, [])
    return [span]


# Variables: Whether to delete target or non-target
# word pool
# option_as_metric = "recall" or "precision"
def get_ps_replace_w_fixed_word_pool(forward_fn,
                                     target_idx,
                                     replacee_span_list,
                                     option_as_metric) -> PSReplace:
    def get_word_pool(e) -> List[List[int]]:
        words = [span.ngram_ids for span in replacee_span_list]
        return words

    eval_metric = get_ps_replace_inner(forward_fn, target_idx, option_as_metric, get_word_pool)
    return eval_metric


def get_ps_replace_100words(interface, target_idx, option) -> PSReplace:
    client = get_mmd_cache_client(interface)
    replacee_span_list = get_100_random_spans()
    return get_ps_replace_w_fixed_word_pool(client.predict, target_idx, replacee_span_list, option)


def get_ps_replace_inner(forward_fn,
                         target_idx,
                         option_as_metric,
                         get_word_pool) -> PSReplace:
    if option_as_metric == "recall":
        pair_modify_fn = get_not_related_pair_replace_fn(target_idx)
        score_combine_fn = discretized_average
    elif option_as_metric == "precision":
        pair_modify_fn = get_related_pair_replace_fn(target_idx)
        score_combine_fn = discretized_average_minus
    else:
        raise ValueError
    eval_metric = PSReplace(forward_fn, pair_modify_fn, get_word_pool, score_combine_fn)
    return eval_metric
