import os
from typing import List

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText, word_segment_w_indices
from cache import save_list_to_jsonl
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.loader import load_mmde_problem


def reform_segment_to_word_level(tokenizer, segment: SegmentedInstance) -> SegmentedInstance:
    text1 = segment.text1
    text2 = segment.text2
    pair = word_segment_w_indices(tokenizer, text1.tokens_ids)
    new_text1 = SegmentedText(*pair)
    return SegmentedInstance(new_text1, text2)


def reform_rei(tokenizer, rei: RelatedEvalInstance) -> RelatedEvalInstance:
    new_seg = reform_segment_to_word_level(tokenizer, rei.seg_instance)
    return RelatedEvalInstance(rei.problem_id, rei.query_info, new_seg, rei.score)


def do_for_dataset(source_dataset, out_name):
    problems: List[RelatedEvalInstance] = load_mmde_problem(source_dataset)
    tokenizer = get_tokenizer()
    new_problems: List[RelatedEvalInstance] = [reform_rei(tokenizer, p) for p in problems]
    save_path = os.path.join(output_path, "qtype", "MMDE_{}_problems.json".format(out_name))
    save_list_to_jsonl(new_problems, save_path)


def main():
    do_for_dataset("dev_sent", "dev_sw")


if __name__ == "__main__":
    main()