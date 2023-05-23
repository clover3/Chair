from typing import Dict, Tuple

import numpy as np

from dataset_specific.mnli.mnli_reader import MNLIReader
from misc_lib import SuccessCounter
from trainer_v2.per_project.tli.token_level_inference import nc_max_e_avg_reduce_then_softmax


def eval_accuracy(tli_module):
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
    acc = suc_counter.get_suc_prob()
    return acc