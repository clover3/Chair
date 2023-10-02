import json
from typing import List, Tuple
from cpath import output_path
from misc_lib import path_join

import numpy as np
import tensorflow as tf

from list_lib import left
from misc_lib import TimeEstimator
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.alignment.galign_inf_helper import build_dataset_q_term_d_term


def fetch_align_probe_all_concat(outputs):
    scores = outputs['align_probe']['all_concat'][:, 0]
    return scores


def get_matching_terms_fn(model, batch_size):
    def get_matching_terms(q_term_id: int) -> List[Tuple[int, float]]:
        d_term_id_st = 1000
        d_term_id_ed = 30000
        eval_dataset = build_dataset_q_term_d_term(q_term_id, d_term_id_st, d_term_id_ed)
        batched_dataset = eval_dataset.batch(batch_size)
        outputs = model.predict(batched_dataset)
        c_log.info("Neural inference done")
        scores = outputs

        # Identify all indices where score is greater than 0
        positive_indices, = np.where(np.less(0, scores))
        results = []

        d_term_list = list(range(d_term_id_st, d_term_id_ed))
        for i in positive_indices:
            d_term = d_term_list[i]
            score = scores[i]
            results.append((d_term, float(score)))


        results.sort(key=lambda x: x[1], reverse=True)
        c_log.info("Return output")
        return results

    return get_matching_terms


def predict_d_terms(freq_q_terms, model, tokenizer, d_term_save_path, job_no):
    # Prepare TF model
    batch_size = 16
    get_matching_terms = get_matching_terms_fn(model, batch_size)
    n_per_job = 100
    st = n_per_job * job_no
    ed = st + n_per_job
    q_term_list = [freq_q_terms[i] for i in range(st, ed)]
    f = open(d_term_save_path, "w")
    ticker = TimeEstimator(len(q_term_list))
    for q_term in q_term_list:
        try:
            q_term_id = tokenizer.convert_tokens_to_ids([q_term])[0]
            over_zero: List[Tuple[int, float]] = get_matching_terms(q_term_id)
            matching_term_indices: List[int] = left(over_zero)
            matching_terms = tokenizer.convert_ids_to_tokens(matching_term_indices)
            c_log.info("Term %s has %d matching entries: %s ", q_term,
                       len(matching_term_indices), str(matching_terms[:10]))
            row = {'q_term': q_term_id, 'matching': over_zero}
            f.write(json.dumps(row) + "\n")
        except KeyError as e:
            print(e)
            print("Skip", q_term)
        ticker.tick()


def build_model_with_output_mapping(model_save_path, mapping_fn):
    model = tf.keras.models.load_model(model_save_path, compile=False)
    model.summary()
    new_outputs = mapping_fn(model.output)
    new_model = tf.keras.models.Model(inputs=model.inputs, outputs=new_outputs)
    return new_model