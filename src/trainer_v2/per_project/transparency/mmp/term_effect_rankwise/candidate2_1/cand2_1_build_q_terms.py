import json

from list_lib import left
from taskman_client.wrapper3 import report_run3
from trainer_v2.per_project.transparency.mmp.alignment.galign_inf_helper import build_dataset_q_term_d_term
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_mmp_galign_path_helper, \
    MMPGAlignPathHelper
import os
from data_generator.tokenizer_wo_tf import get_tokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cpath import output_path
from misc_lib import path_join

import sys
from trainer_v2.chair_logging import c_log
import tensorflow as tf
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np


def get_matching_terms_fn(batch_size, model, top_k):
    def get_matching_terms(q_term_id: int) -> List[Tuple[int, float]]:
        d_term_id_st = 1000
        d_term_id_ed = 3000
        eval_dataset = build_dataset_q_term_d_term(q_term_id, d_term_id_st, d_term_id_ed)
        batched_dataset = eval_dataset.batch(batch_size)
        outputs = model.predict(batched_dataset)
        scores = outputs['align_probe']['all_concat'][:, 0]

        # Get the indices of top-k scores
        top_k_indices = np.argpartition(scores, len(scores) - top_k)[-top_k:]

        # Identify all indices where score is greater than 0
        positive_indices, = np.where(np.less(0, scores))

        # Combine top-k and positive indices
        if len(positive_indices) > 0:
            combined_indices = np.concatenate([top_k_indices, positive_indices])
            combined_indices = np.unique(combined_indices)
        else:
            combined_indices = top_k_indices

        combined_indices = np.unique(combined_indices)
        results = []

        for i, record in enumerate(eval_dataset):
            if i in combined_indices:
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
    tokenizer = get_tokenizer()

    model_save_path = sys.argv[1]
    job_no = int(sys.argv[2])
    model = tf.keras.models.load_model(model_save_path, compile=False)
    print("Job no:", job_no)

    # Select 1K query terms that are frequent
    config: MMPGAlignPathHelper = get_mmp_galign_path_helper()
    freq_q_terms = config.load_freq_q_terms()
    batch_size = 16
    top_k = 10
    get_matching_terms = get_matching_terms_fn(batch_size, model, top_k)

    n_per_job = 100
    st = n_per_job * job_no
    ed = st + n_per_job
    save_path = path_join(output_path, "msmarco", "passage", "candidate2_1_building", f"{job_no}.jsonl")
    f = open(save_path, "w")

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


if __name__ == "__main__":
    main()