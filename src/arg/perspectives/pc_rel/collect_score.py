from typing import Dict

from misc_lib import TimeEstimator
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def collect_pc_rel_score(prediction_file, info: Dict):
    data = EstimatorPredictionViewer(prediction_file)

    print("Num data ", data.data_len)
    group_by_key = {}
    num_append = 0
    last_claim = None
    ticker = TimeEstimator(data.data_len)
    for entry in data:
        ticker.tick()
        logits = entry.get_vector("logits")
        data_id = entry.get_vector("data_id")[0]
        try:
            cur_info = info[data_id]
            if 'cid' in cur_info:
                cid = cur_info['cid']
                last_claim = cid, logits
            elif 'pid' in cur_info:
                pid = cur_info['pid']
                cid, c_logits = last_claim
                key = cid, pid
                if key not in group_by_key:
                    group_by_key[key] = []
                group_by_key[key].append((c_logits, logits))
                num_append += 1
            else:
                assert False
        except KeyError as e:
            print(e)
            pass
    print(num_append)
    return group_by_key
