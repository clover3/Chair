
from typing import Dict, Tuple, List, NewType

from scipy.special import softmax

from arg.perspectives.pc_rel.collect_score import collect_pc_rel_score
from cache import load_from_pickle, save_to_pickle
from cpath import pjoin, output_path

CPIDPair = NewType('CPIDPair', Tuple[int, int])
Logits = NewType('Logits', List[float])


def collect_save_relevance_score(prediction_path, pc_rel_info):
    info_d = load_from_pickle(pc_rel_info)
    print('info_d',  len(info_d))

    relevance_scores: Dict[CPIDPair, List[Tuple[Logits, Logits]]] = collect_pc_rel_score(prediction_path, info_d)

    output_score_dict = {}

    for key in relevance_scores:
        scores: List[Tuple[List[float], List[float]]] = relevance_scores[key]
        print(key, len(scores))
        pc_count = 0
        for c_logits, p_logits in scores:
            c_rel = softmax(c_logits)[1] > 0.5
            p_rel = softmax(p_logits)[1] > 0.5
            pc_count += int(c_rel and p_rel)

        cid, pid = key
        cpid = "{}_{}".format(cid, pid)
        output_score_dict[cpid] = pc_count
    print(len(output_score_dict))
    return output_score_dict


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
