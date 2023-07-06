import os

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_deep_model_score_path_train, \
    get_deep_model_score_path


def get_valid_mmp_partition_for_train():
    yield from range(0, 109)
    yield from range(113, 119)


def get_valid_mmp_partition_for_dev():
    yield from range(0, 111)


def get_valid_mmp_partition(split):
    if split == "train":
        return get_valid_mmp_partition_for_train()
    elif split == "dev":
        return get_valid_mmp_partition_for_dev()
    else:
        raise ValueError()


def get_mmp_split_w_deep_scores_train():
    split = "train"
    return get_mmp_split_w_deep_scores(split)


def get_mmp_split_w_deep_scores(split):
    all_valid = list(get_valid_mmp_partition(split))
    has_deep_score = []
    for i in all_valid:
        check_path = get_deep_model_score_path(split, i)
        if os.path.exists(check_path):
            has_deep_score.append(i)
    return has_deep_score