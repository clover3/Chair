from typing import Dict, Tuple

from arg.perspectives.types import DataID, CPIDPair
from arg.qck.decl import qck_convert_map, qc_convert_map
from arg.qck.prediction_reader import parse_info_inner
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def collect_scores(prediction_file, info: Dict, logit_to_score) \
        -> Dict[DataID, Tuple[CPIDPair, float]]:
    data = EstimatorPredictionViewer(prediction_file)
    print("Num data ", data.data_len)
    out_d: Dict[DataID, Tuple[CPIDPair, float]] = {}
    for entry in data:
        logits = entry.get_vector("logits")
        score = logit_to_score(logits)
        data_id = entry.get_vector("data_id")[0]
        try:
            cur_info = info[str(data_id)]

            if 'kdp' in cur_info:
                parse_info_inner(cur_info, qck_convert_map, True)
                cid = int(cur_info['query'].query_id)
                pid = int(cur_info['candidate'].id)
            elif 'query' in cur_info:
                parse_info_inner(cur_info, qc_convert_map, True)
                cid = int(cur_info['query'].query_id)
                pid = int(cur_info['candidate'].id)
            else:
                cid = cur_info['cid']
                pid = cur_info['pid']
            cpid = CPIDPair((cid, pid))
            out_d[data_id] = (cpid, score )
        except KeyError as e:
            print("Key error", e)
            print("data_id", data_id)
            pass
    return out_d


