import os
import random
from typing import List, Callable

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from cache import save_list_to_jsonl
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from misc_lib import exist_or_mkdir, tprint, TEL
from tlm.qtype.analysis_fde.fde_module import get_fde_module
from tlm.qtype.partial_relevance.attention_based.mmd_z_client import get_mmd_client_wrap
from tlm.qtype.partial_relevance.complement_search_pckg.check import CheckComplementCandidate
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementCandidateGenIF, \
    PartialSegment
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.complement_search_pckg.original_query import ComplementGenOriginalQuery
from tlm.qtype.partial_relevance.complement_search_pckg.query_vector import ComplementGenByQueryVector
from tlm.qtype.partial_relevance.complement_search_pckg.span_iter import ComplementGenBySpanIter
from contradiction.alignment.data_structure.print_helper import rei_to_text
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.loader import load_mmde_problem


def get_complement_checker():
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client_wrap()
    return CheckComplementCandidate(forward_fn, FuncContentSegJoinPolicy())


def run_complement_search_loop(problems: List[RelatedEvalInstance],
                               candi_gen: ComplementCandidateGenIF,
                               preserve_seg_idx: int,
                               max_candidates = 100
                               ) -> List[ComplementSearchOutput]:
    checker: CheckComplementCandidate = get_complement_checker()
    if preserve_seg_idx == 0:
        tprint("Preserve functional span , search for content span")
    elif preserve_seg_idx == 1:
        tprint("Preserve content span , search for functional span")

    tokenizer = get_tokenizer()
    cs_output_list: List[ComplementSearchOutput] = []
    for p in TEL(problems):
        tprint(rei_to_text(tokenizer, p))
        keep_seg_text = ids_to_text(tokenizer, p.seg_instance.text1.get_tokens_for_seg(preserve_seg_idx))
        tprint("Look for complement of [{}]".format(keep_seg_text))
        candidate_complement_list: List[PartialSegment] = candi_gen.get_candidates(p.seg_instance, preserve_seg_idx)
        if len(candidate_complement_list) > max_candidates:
            tprint("{} candidates".format(len(candidate_complement_list)))
            candidate_complement_list = random.sample(candidate_complement_list, max_candidates)
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


def get_candidate_generator_by_name(method) -> ComplementCandidateGenIF:
    if method == "span_iter":
        candidate_generator = ComplementGenBySpanIter()
    elif method == "original_query":
        candidate_generator = ComplementGenOriginalQuery()
    elif method == "q_vector":
        candidate_generator = ComplementGenByQueryVector(get_fde_module())
    else:
        raise ValueError()

    return candidate_generator


def run_search_and_save(dataset, method, preserve_seg_idx):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    candidate_generator: ComplementCandidateGenIF = get_candidate_generator_by_name(method)
    cs_output_list = run_complement_search_loop(problems, candidate_generator, preserve_seg_idx)
    run_name = f"{dataset}_{method}_{preserve_seg_idx}"
    save_common(cs_output_list, run_name)


def main():
    preserve_seg_idx = 0
    dataset = "dev_sm"
    # run_search_and_save(dataset, "span_iter", preserve_seg_idx)
    run_search_and_save(dataset, "original_query", preserve_seg_idx)


if __name__ == "__main__":
    main()
