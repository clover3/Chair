from typing import Dict, Tuple
from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference, nc_max_e_avg_reduce_then_softmax

import logging
import numpy as np

from misc_lib import SuccessCounter
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from dataset_specific.mnli.mnli_reader import MNLIReader
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import get_pep_cache_client


def main():
    c_log.setLevel(logging.DEBUG)
    nli_predict_fn = get_pep_cache_client()
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)

    reader = MNLIReader()
    nli_pairs = list(reader.get_dev())
    tli_payload = []
    for pair in nli_pairs:
        tli_payload.append((pair.premise, pair.hypothesis))

    tli_output_list = tli_module.do_batch(tli_payload)
    tli_dict: Dict[Tuple[str, str], np.array] = dict(zip(tli_payload, tli_output_list))
    suc_counter = SuccessCounter()
    for pair in nli_pairs:
        prem = pair.premise
        hypo = pair.hypothesis
        tli: np.array = tli_dict[prem, hypo]
        probs_from_tli = nc_max_e_avg_reduce_then_softmax(tli)
        pred = np.argmax(probs_from_tli)
        is_correct = pred == pair.get_label_as_int()
        suc_counter.add(is_correct)

    print("Accuracy: ", suc_counter.get_suc_prob())




if __name__ == "__main__":
    main()