from typing import List, Callable

from tlm.qtype.partial_relevance.complement_search_pckg.complement_candidate_gen_if import ComplementCandidateGenIF, \
    PartialSegment

from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from tlm.qtype.analysis_fde.fde_module import get_fde_module
from tlm.qtype.partial_relevance.attention_based.mmd_z_client import get_mmd_client_wrap
from tlm.qtype.partial_relevance.complement_search_pckg.check import CheckComplementCandidate
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.complement_search_pckg.query_vector import ComplementGenByQueryVector
from tlm.qtype.partial_relevance.complement_search_pckg.span_iter import ComplementGenBySpanIter
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, SegmentedInstance, rei_to_text
from tlm.qtype.partial_relevance.loader import load_dev_small_problems


def get_complement_checker():
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client_wrap()
    return CheckComplementCandidate(forward_fn, FuncContentSegJoinPolicy())


def run_test_search(problems: List[RelatedEvalInstance],
                    candi_gen: ComplementCandidateGenIF,
                    preserve_seg_idx: int):
    checker: CheckComplementCandidate = get_complement_checker()
    if preserve_seg_idx == 0:
        print("Preserve functional span , search for content span")
    elif preserve_seg_idx == 1:
        print("Preserve content span , search for functional span")

    tokenizer = get_tokenizer()
    for p in problems:
        print(rei_to_text(tokenizer, p))
        keep_seg_text = ids_to_text(tokenizer, p.seg_instance.text1.get_tokens_for_seg(preserve_seg_idx))
        print("Look for complement of [{}]".format(keep_seg_text))
        candidate_complement_list: List[PartialSegment] = candi_gen.get_candidates(p.seg_instance, preserve_seg_idx)

        valid_complement_list: List[PartialSegment] = checker.check_complement_list(p.seg_instance, preserve_seg_idx,
                                                              candidate_complement_list)
        valid_complement_list_s: List[str] = [c.to_text(tokenizer) for c in valid_complement_list]
        if valid_complement_list_s:
            print("Found: ", valid_complement_list_s)
        else:
            print("Not found matching complement")
        print("----")


def run_search_with_query_vector():
    preserve_seg_idx = 1
    problems: List[RelatedEvalInstance] = load_dev_small_problems()
    candidate_generator = ComplementGenByQueryVector(get_fde_module())
    run_test_search(problems, candidate_generator, preserve_seg_idx)


def run_search_with_span_iter():
    preserve_seg_idx = 1
    problems: List[RelatedEvalInstance] = load_dev_small_problems()
    candidate_generator = ComplementGenBySpanIter()
    run_test_search(problems, candidate_generator, preserve_seg_idx)


def main():
    run_search_with_query_vector()


if __name__ == "__main__":
    main()
