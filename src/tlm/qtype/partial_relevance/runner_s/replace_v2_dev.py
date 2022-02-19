from typing import List

from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedBinaryAnswer
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import get_100_random_spans
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import TenStepRandomReplacePolicy
from tlm.qtype.partial_relevance.eval_metric_binary.eval_conditional_v2_single_target import \
    conditional_align_eval_replace
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v2 import ReplaceV2SingleSeg
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.mmd_cached_client import get_mmd_cache_client
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_binary_related_eval_answer


def run_align_eval_b(dataset, method, interface="localhost"):
    client = get_mmd_cache_client(interface)
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset, method)
    replacee_span_list = get_100_random_spans()
    def get_word_pool(e) -> List[List[int]]:
        words = [span.ngram_ids for span in replacee_span_list]
        return words

    target_seg_idx = 1
    eval_policy = ReplaceV2SingleSeg(client.predict, TenStepRandomReplacePolicy(), target_seg_idx, get_word_pool)
    items = conditional_align_eval_replace(answers, problems, eval_policy)
    print(items)


if __name__ == "__main__":
    run_align_eval_b("dev_sent", "exact_match")