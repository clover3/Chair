import random
from typing import List, Callable
import os
from cpath import output_path
from misc_lib import exist_or_mkdir, tprint, TEL
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementCandidateGenIF, \
    PartialSegment

from cache import save_list_to_jsonl
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from tlm.qtype.analysis_fde.fde_module import get_fde_module
from tlm.qtype.partial_relevance.attention_based.mmd_z_client import get_mmd_client_wrap
from tlm.qtype.partial_relevance.complement_search_pckg.check import CheckComplementCandidate
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.complement_search_pckg.original_query import ComplementGenOriginalQuery
from tlm.qtype.partial_relevance.complement_search_pckg.query_vector import ComplementGenByQueryVector
from tlm.qtype.partial_relevance.complement_search_pckg.span_iter import ComplementGenBySpanIter
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, SegmentedInstance, rei_to_text
from tlm.qtype.partial_relevance.loader import load_dev_small_problems, load_dev_problems


def get_complement_checker():
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client_wrap()
    return CheckComplementCandidate(forward_fn, FuncContentSegJoinPolicy())


def run_test_search(problems: List[RelatedEvalInstance],
                    candi_gen: ComplementCandidateGenIF,
                    preserve_seg_idx: int) -> List[ComplementSearchOutput]:
    checker: CheckComplementCandidate = get_complement_checker()
    if preserve_seg_idx == 0:
        tprint("Preserve functional span , search for content span")
    elif preserve_seg_idx == 1:
        tprint("Preserve content span , search for functional span")

    tokenizer = get_tokenizer()
    cs_output_list: List[ComplementSearchOutput] = []
    for p in TEL(problems):
        # tprint(rei_to_text(tokenizer, p))
        keep_seg_text = ids_to_text(tokenizer, p.seg_instance.text1.get_tokens_for_seg(preserve_seg_idx))
        tprint("Look for complement of [{}]".format(keep_seg_text))
        candidate_complement_list: List[PartialSegment] = candi_gen.get_candidates(p.seg_instance, preserve_seg_idx)
        if len(candidate_complement_list) > 100:
            tprint("{} candidates".format(len(candidate_complement_list)))
            candidate_complement_list = random.sample(candidate_complement_list, 100)
        tprint("{} candidates".format(len(candidate_complement_list)))
        valid_complement_list: List[PartialSegment] = checker.check_complement_list(p.seg_instance, preserve_seg_idx,
                                                              candidate_complement_list)
        valid_complement_list_s: List[str] = [c.to_text(tokenizer) for c in valid_complement_list]
        cs_output = ComplementSearchOutput(p.problem_id, preserve_seg_idx, valid_complement_list)
        cs_output_list.append(cs_output)
        if valid_complement_list_s:
            tprint("Found: ", valid_complement_list_s)
        else:
            tprint("Not found matching complement")
        tprint("----")
    return cs_output_list


def save_common(cs_output_list, run_name):
    save_dir = os.path.join(output_path, "qtype", "comp_search")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, "{}.jsonl".format(run_name))
    save_list_to_jsonl(cs_output_list, save_path)


def run_search_with_query_vector():
    preserve_seg_idx = 1
    problems: List[RelatedEvalInstance] = load_dev_problems()
    candidate_generator = ComplementGenByQueryVector(get_fde_module())
    cs_output_list = run_test_search(problems, candidate_generator, preserve_seg_idx)
    save_common(cs_output_list, "q_vector")


def run_search_with_span_iter():
    preserve_seg_idx = 1
    problems: List[RelatedEvalInstance] = load_dev_problems()
    candidate_generator = ComplementGenBySpanIter()
    cs_output_list = run_test_search(problems, candidate_generator, preserve_seg_idx)
    save_common(cs_output_list, "span_iter")


def run_search_with_original_query():
    preserve_seg_idx = 1
    problems: List[RelatedEvalInstance] = load_dev_problems()
    candidate_generator = ComplementGenOriginalQuery()
    cs_output_list = run_test_search(problems, candidate_generator, preserve_seg_idx)
    save_common(cs_output_list, "original_query")


def main():
    run_search_with_original_query()


if __name__ == "__main__":
    main()
