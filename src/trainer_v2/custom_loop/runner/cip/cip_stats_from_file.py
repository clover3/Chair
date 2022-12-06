import os
from typing import List, Iterator

from cache import load_list_from_jsonl
from trainer_v2.custom_loop.per_task.cip.cip_common import SegmentationTrials, \
    Prediction, Comparison, get_statistics
from trainer_v2.custom_loop.per_task.cip.path_helper import get_nli_baseline_pred_save_path, \
    get_nlits_segmentation_trial_subjob_save_dir


def iter_cip_preds() -> Iterator[Comparison]:
    split = "train"
    dir_path = get_nlits_segmentation_trial_subjob_save_dir()

    def iter_seg_trials() -> Iterator[SegmentationTrials]:
        for job_no in range(80):
            jsonl_path = os.path.join(dir_path, "nlits_trials", str(job_no))
            st_list: List[SegmentationTrials] = load_list_from_jsonl(jsonl_path, SegmentationTrials.from_json)
            yield from st_list

    def iter_base_outputs() -> Iterator[Prediction]:
        yield from load_list_from_jsonl(get_nli_baseline_pred_save_path(split), Prediction.from_json)

    for base, seg_trials in zip(iter_base_outputs(), iter_seg_trials()):
        yield Comparison(base.prem, base.hypo, base.label, base.pred, seg_trials.seg_outputs, seg_trials.st_ed_list)


def main():
    get_statistics(iter_cip_preds())


if __name__ == "__main__":
    main()
