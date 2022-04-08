
import copy
from typing import List, Tuple

from bert_api.segmented_instance.segmented_text import SegmentedText
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from contradiction.alignment.data_structure.related_eval_instance import RelatedEvalInstance


def backtracking(deletion_array: List[bool], max_segment, is_flip) -> List[Tuple[List[bool], List[bool]]]:
    output = []
    # Search for decision flip
    n_item = len(deletion_array)
    if n_item == max_segment:
        return []

    for new_decision in [True, False]:
        new_array = copy.deepcopy(deletion_array)
        new_array.append(new_decision)
        if is_flip(new_array):
            flip_point = deletion_array, new_array
            output.append(flip_point)

        ret = backtracking(new_array, max_segment, is_flip)
        output.extend(ret)

    return output


def get_seg_deleted(text: SegmentedText, deletion_array) -> SegmentedText:
    pass


def outer_loop(problem: RelatedEvalInstance, complement: ComplementSearchOutput):
    # TODO
    #    Given n document segments
    #    Execute all 2^n combinations of deletion.
    #    D = Set of all sub-documents
    #    PartialRelevance(d_i)
    #    Select a complement c
    #
    if not complement.complement_list:
        return

    def get_score(new_text: SegmentedText):
        pass

    n_segment = problem.seg_instance.text2.get_seg_len()
    first_complement = complement.complement_list[0]

    score_cache = {}
    def is_flip(deletion_array):
        if not deletion_array[-1]:
            # deletion_array[:-1] should have same results as deletion_array
            score = score_cache[str(deletion_array)]
        else:
            new_text: SegmentedText = get_seg_deleted(problem.seg_instance.text2, deletion_array)
            score = get_score(new_text)
        return score < 0.5

    backtracking([], n_segment, is_flip)