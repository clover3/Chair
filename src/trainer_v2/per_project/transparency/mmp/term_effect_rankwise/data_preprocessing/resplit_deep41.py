from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.data_preprocessing.resplit_scores import save_tsv
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_deep_score_save_path_by_qid
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure_mmp import load_deep_scores


def main():
    job_no = 41
    deep_score_grouped: List[List] = load_deep_scores("train", job_no)
    for group in deep_score_grouped:
        qid, _, _ = group[0]
        save_path = get_deep_score_save_path_by_qid(qid)
        save_tsv(group, save_path)


if __name__ == "__main__":
    main()