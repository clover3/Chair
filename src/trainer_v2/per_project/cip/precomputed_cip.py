import itertools
import os
from typing import Iterator, List

from cache import load_list_from_jsonl
from trainer_v2.per_project.cip.cip_common import Comparison, SegmentationTrials, Prediction
from trainer_v2.per_project.cip.path_helper import get_nlits_segmentation_trial_subjob_save_dir, \
    get_nli_baseline_pred_save_path


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


def get_cip_pred_splits_iter():
    k_validation = 4000
    iter: Iterator[Comparison] = iter_cip_preds()
    val_itr = itertools.islice(iter, k_validation)
    train_itr = itertools.islice(iter, k_validation, None)
    n_train_itr_size = 384702
    todo = [
        ("train_val", val_itr, k_validation),
        ("train", train_itr, n_train_itr_size)
    ]
    return todo
