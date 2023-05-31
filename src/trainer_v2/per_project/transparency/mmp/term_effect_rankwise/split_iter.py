import os

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_deep_model_score_path


def get_valid_mmp_split():
    yield from range(0, 109)
    yield from range(113, 119)


def get_mmp_split_w_deep_scores():
    all_valid = list(get_valid_mmp_split())
    has_deep_score = []
    for i in all_valid:
        check_path = get_deep_model_score_path(i)
        if os.path.exists(check_path):
            has_deep_score.append(i)
    return has_deep_score