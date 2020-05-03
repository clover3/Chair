from typing import Dict, List

from arg.perspectives.cpid_def import CPID
from arg.perspectives.pc_rel.scorer import collect_pipeline2_score
from cache import save_to_pickle
from cpath import output_path, pjoin
from list_lib import dict_value_map


def save_dev():
    prediction_path = pjoin(output_path, "tf_rel_filter_B_dev")
    scores: Dict[CPID, List[float]] = collect_pipeline2_score(prediction_path, "pc_rel_dev_info_all")
    reduced_score: Dict[CPID, float] = dict_value_map(sum, scores)
    save_to_pickle(reduced_score, "tf_rel_filter_B_dev_score")


def main():
    save_dev()

if __name__ == "__main__":
    main()