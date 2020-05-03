

from arg.perspectives.pc_rel.scorer import collect_save_relevance_score
from cache import save_to_pickle
from cpath import pjoin, output_path

def save_train():
    prediction_path = pjoin(output_path, "pc_rel")
    pc_rel_based_score = collect_save_relevance_score(prediction_path , "pc_rel_info_all")
    save_to_pickle(pc_rel_based_score , "pc_rel_based_score_train")


def save_dev():
    prediction_path = pjoin(output_path, "pc_rel_dev")
    pc_rel_based_score = collect_save_relevance_score(prediction_path , "pc_rel_dev_info_all")
    save_to_pickle(pc_rel_based_score, "pc_rel_based_score_dev")


if __name__ == "__main__":
    save_dev()

