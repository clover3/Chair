import json
import os

from alignment import RelatedEvalInstance
from bert_api import SegmentedText, SegmentedInstance
from cache import load_list_from_jsonl, save_list_to_jsonl_w_fn
from cpath import output_path


def main():
    src_path = os.path.join(output_path, "qtype", "MMDE_dev_sent_problems.json")
    item = load_list_from_jsonl(src_path, lambda x: x)
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_sent_problems_fixed.json")

    def transform(e):
        problem_id, seg_instance_raw, score = e
        text1_raw, text2_raw = seg_instance_raw

        seg_instance = SegmentedInstance(SegmentedText(*text1_raw), SegmentedText(*text2_raw))
        e: RelatedEvalInstance = RelatedEvalInstance(problem_id, seg_instance, score)
        return e

    save_list_to_jsonl_w_fn(map(transform, item), save_path, RelatedEvalInstance.to_json)


if __name__ == "__main__":
    main()