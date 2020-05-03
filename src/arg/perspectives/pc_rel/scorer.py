
from typing import Dict, Tuple, List

from scipy.special import softmax

from arg.perspectives.cpid_def import CPID
from arg.perspectives.pc_rel.collect_score import collect_pc_rel_score
from arg.perspectives.types import CPIDPair, Logits
from cache import load_from_pickle
from misc_lib import TimeEstimator
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def collect_save_relevance_score(prediction_path, pc_rel_info) -> Dict[str, int]:
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


def collect_pipeline2_score(prediction_path, pc_rel_info) -> Dict[CPID, List[float]]:
    info_d = load_from_pickle(pc_rel_info)
    print('info_d',  len(info_d))

    def get_cpid(data_id, info_d) -> CPID:
        info_1 = info_d[data_id-1]
        info_2 = info_d[data_id]
        cid = info_1['cid']
        pid = info_2['pid']
        return CPID("{}_{}".format(cid, pid))

    data = EstimatorPredictionViewer(prediction_path)

    print("Num data ", data.data_len)
    ticker = TimeEstimator(data.data_len)
    score_list_d : Dict[CPID, List] = {}
    for entry in data:
        ticker.tick()
        logits = entry.get_vector("logits")
        probs = softmax(logits)
        score = probs[1]
        data_id = entry.get_vector("data_id")[0]

        cpid: CPID = get_cpid(data_id, info_d)

        if cpid not in score_list_d:
            score_list_d[cpid] = []
        score_list_d[cpid].append(score)

    return score_list_d