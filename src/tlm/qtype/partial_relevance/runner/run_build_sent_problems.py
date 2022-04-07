import os
from typing import Callable, List

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import word_segment_w_indices
from cache import save_list_to_jsonl
from cpath import output_path
from epath import job_man_dir
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.problem_builder import build_sentence_as_doc
from tlm.qtype.partial_relevance.runner.run_eval_old.run_partial_related_full_eval import get_mmd_client
from tlm.qtype.partial_relevance.runner.sent_tokenize_dev import sentence_segment_w_indices


def build():
    info_path = os.path.join(job_man_dir, "MMDE_dev_info")
    raw_prediction_path = os.path.join(output_path, "qtype", "MMDE_dev_mmd_Z.score")
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client("localhost")
    problems_wo_scores: List[RelatedEvalInstance] = build_sentence_as_doc(info_path, raw_prediction_path,
                                  sentence_segment_w_indices,
                                  word_segment_w_indices, 1000)

    scores = forward_fn([p.seg_instance for p in problems_wo_scores])

    new_items: List[RelatedEvalInstance] = []
    for old_p, score in zip(problems_wo_scores, scores):
        if score >= 0.5:
            new_p = RelatedEvalInstance(
                old_p.problem_id,
                old_p.query_info,
                old_p.seg_instance,
                score)
            new_items.append(new_p)

    print("{} sentences -> {} problems".format(len(problems_wo_scores), len(new_items)))
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_sent_problems.json")
    save_list_to_jsonl(new_items, save_path)


def main():
    build()


if __name__ == "__main__":
    main()
