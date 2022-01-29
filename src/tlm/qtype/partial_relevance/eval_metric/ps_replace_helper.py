from typing import NamedTuple, List

from cache import load_pickle_from
from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import PartialSegment
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedText, SegmentedInstance, RelatedEvalInstance, \
    RelatedEvalAnswer
from tlm.qtype.partial_relevance.eval_metric.doc_modify_fns import DocReplaceFunc, get_replace_zero, \
    get_replace_non_zero
from tlm.qtype.partial_relevance.eval_metric.ps_replace import PSReplace


class RandomSpan(NamedTuple):
    ngram: List[str]
    tag: str
    idf_sum: float
    ngram_ids: List[int]

    def __str__(self):
        return "RandomSpan({0}, {1}, {2:.2f})".format(" ".join(self.ngram), self.tag, self.idf_sum)

    def get_ngram_s(self):
        return " ".join(self.ngram)


def load_random_span() -> List[RandomSpan]:
    save_path = at_output_dir("qtype", "msmarco_random_spans.pickle")
    data = load_pickle_from(save_path)
    tokenizer = get_tokenizer()
    def parse_entry(e) -> RandomSpan:
        ngram_s, tag, idf_sum = e
        ngram_ids: List[int] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join(ngram_s)))
        return RandomSpan(ngram_s, tag, idf_sum, ngram_ids)

    outputs: List[RandomSpan] = []
    for list_per_doc in data:
        outputs.extend(map(parse_entry, list_per_doc))
    return outputs


def get_100_random_spans():
    random_spans: List[RandomSpan] = load_random_span()[:100]
    return random_spans


def get_pair_modify_fn(query_modify_fn,
                       doc_modify_fn,
                       target_seg_idx):
    def pair_modify_fn(answer: RelatedEvalAnswer,
                       problem: RelatedEvalInstance,
                       word: List[int]):

        doc_term_scores: List[float] = answer.contribution.table[target_seg_idx]
        new_doc: SegmentedText = doc_modify_fn(problem.seg_instance.text2,
                                               doc_term_scores, word)
        full_query = problem.seg_instance.text1
        new_query: SegmentedText = query_modify_fn(full_query, target_seg_idx, word)
        seg = SegmentedInstance(new_query, new_doc)
        return seg
    return pair_modify_fn


# Variables: Whether to delete target or non-target
# word pool
def get_metric_ps_replace(forward_fn, target_idx, replace_non_target=True) -> PSReplace:
    random_spans = get_100_random_spans()

    def get_word_pool(e) -> List[List[int]]:
        words = [span.ngram_ids for span in random_spans]
        return words

    join_policy = FuncContentSegJoinPolicy()

    def query_modify_fn(query: SegmentedText, word):
        if replace_non_target:
            preserve_idx = target_idx
        else:  # Remove target
            preserve_idx = 1 - target_idx
        partial_tokens = PartialSegment.init_one_piece(word)
        new_query: SegmentedText = join_policy.join_tokens(query, partial_tokens, preserve_idx)
        return new_query

    if replace_non_target:
        doc_modify_fn: DocReplaceFunc = get_replace_zero()
    else:
        doc_modify_fn: DocReplaceFunc = get_replace_non_zero()
    pair_modify_fn = get_pair_modify_fn(query_modify_fn, doc_modify_fn, target_idx)

    eval_metric = PSReplace(forward_fn, pair_modify_fn, get_word_pool)
    return eval_metric

