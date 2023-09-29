import json

from list_lib import left
from trainer_v2.per_project.transparency.mmp.alignment.galign_inf_helper import build_dataset_q_term_d_term
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate2_2.path_helper import \
    get_candidate2_2_term_pair_candidate_building_path
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignPathHelper, get_cand2_2_path_helper
import os
from data_generator.tokenizer_wo_tf import get_tokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from misc_lib import TimeEstimator

import sys
from trainer_v2.chair_logging import c_log
import tensorflow as tf
from typing import List, Tuple
import numpy as np


def get_matching_terms_fn(model_save_path, batch_size):
    model = tf.keras.models.load_model(model_save_path, compile=False)
    def get_matching_terms(q_term_id: int) -> List[Tuple[int, float]]:
        d_term_id_st = 1000
        d_term_id_ed = 30000
        eval_dataset = build_dataset_q_term_d_term(q_term_id, d_term_id_st, d_term_id_ed)
        batched_dataset = eval_dataset.batch(batch_size)
        outputs = model.predict(batched_dataset)
        scores = outputs['align_probe']['all_concat'][:, 0]

        # Get the indices of top-k scores
        # Identify all indices where score is greater than 0
        positive_indices, = np.where(np.less(0, scores))

        results = []
        for i, record in enumerate(eval_dataset):
            if i in positive_indices:
                d_term = record['d_term'][0].numpy().tolist()
                score = scores[i]
                results.append((d_term, float(score)))

        # Sort the terms by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    return get_matching_terms


# @report_run3
def main():
    c_log.info(__file__)
    model_save_path = sys.argv[1]
    job_no = int(sys.argv[2])

    tokenizer = get_tokenizer()
    print("Job no:", job_no)
    config: MMPGAlignPathHelper = get_cand2_2_path_helper()
    save_path = get_candidate2_2_term_pair_candidate_building_path(job_no)
    freq_q_terms = config.load_freq_q_terms()

    # Prepare TF model
    batch_size = 16
    get_matching_terms = get_matching_terms_fn(model_save_path, batch_size)

    n_per_job = 100
    st = n_per_job * job_no
    ed = st + n_per_job
    f = open(save_path, "w")

    ticker = TimeEstimator(n_per_job)
    for i in range(st, ed):
        q_term: str = freq_q_terms[i]
        try:
            q_term_id = tokenizer.convert_tokens_to_ids([q_term])[0]
            top_scores: List[Tuple[int, float]] = get_matching_terms(q_term_id)
            over_zero = [(term, score) for term, score in top_scores if score > 0]
            matching_term_indices = left(over_zero)
            matching_terms = tokenizer.convert_ids_to_tokens(matching_term_indices)
            c_log.info("Term %s has %d matching entries: %s ", q_term,
                len(matching_term_indices), str(matching_terms[:10]))
            row = {'q_term': q_term_id, 'matching': top_scores }
            f.write(json.dumps(row) + "\n")
        except KeyError as e:
            print("Skip", q_term)
        ticker.tick()


if __name__ == "__main__":
    main()