import os
import pickle
from typing import List, Callable, OrderedDict, Iterator

from cpath import output_path
from data_generator2.segmented_enc.es.common import get_evidence_pred_encode_fn
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.evidence_selector.evidence_scoring import cross_entropy, length_loss
from trainer_v2.evidence_selector.evidence_candidates import ScoredEvidencePair, EvidencePair
import numpy as np


def iter_all_candidate_grouped() -> Iterator[List[ScoredEvidencePair]]:
    for job_id in range(40):
        src_path = os.path.join(job_man_dir, "evidence_candidate_calc", str(job_id))
        try:
            records: List[List[ScoredEvidencePair]] = pickle.load(open(src_path, "rb"))

            records_valid = [group for group in records if len(group) > 1]
            yield from records_valid
        except FileNotFoundError as e:
            print("Skip Job {}".format(job_id))




def get_train_item(prediction_loss, group: List[ScoredEvidencePair]) -> EvidencePair:
    base_item = group[0]
    others = group[1:]
    if not others:
        print("No other", base_item)
    best_prem_i = []
    for prem_i in [0, 1]:
        def evidence_score(item: ScoredEvidencePair) -> float:
            pair = item.pair
            del_indices = [pair.p_del_indices1, pair.p_del_indices2]
            err = prediction_loss(base_item.l_y[prem_i], item.l_y[prem_i])
            l_loss = length_loss(len(del_indices[prem_i]), len(pair.p_tokens))
            tolerance = 0.05
            combined_score = max(tolerance, err) + tolerance * l_loss
            return combined_score

        others.sort(key=evidence_score)

        best: ScoredEvidencePair = others[0]
        best_prem_i.append(best)

    train_item = EvidencePair(
        base_item.pair.p_tokens,
        base_item.pair.h1,
        base_item.pair.h2,
        best_prem_i[0].pair.p_del_indices1,
        best_prem_i[1].pair.p_del_indices2,
    )
    return train_item


def main():
    segment_len = 300
    records = iter_all_candidate_grouped()
    prediction_loss: Callable[[np.array, np.array], float] = cross_entropy

    def get_train_item_fn(group: List[ScoredEvidencePair]):
        return get_train_item(prediction_loss, group)
    train_iter: Iterator[EvidencePair] = map(get_train_item_fn, records)
    encode_fn: Callable[[EvidencePair], OrderedDict] = get_evidence_pred_encode_fn(segment_len)
    output_dir = os.path.join(output_path, "align", "evidence_prediction")
    exist_or_mkdir(output_dir)
    save_path = os.path.join(output_dir, "train")
    write_records_w_encode_fn(save_path, encode_fn, train_iter)



if __name__ == "__main__":
    main()