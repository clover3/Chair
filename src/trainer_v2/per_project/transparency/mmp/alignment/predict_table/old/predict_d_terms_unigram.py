import json

import tensorflow as tf

from typing import List, Callable, Tuple


from trainer_v2.per_project.transparency.mmp.alignment.predict_table.common import predict_d_terms, \
    get_matching_terms_fn



def predict_d_terms_for_job(q_term_list, model, tokenizer, d_term_save_path):
    # Prepare TF model
    batch_size = 16
    get_matching_terms: Callable[[int], List[Tuple[int, float]]] = get_matching_terms_fn(model, batch_size)
    f = open(d_term_save_path, "w")
    out_itr = predict_d_terms(get_matching_terms, q_term_list, tokenizer)
    for row in out_itr:
        f.write(json.dumps(row) + "\n")


def build_model_with_output_mapping(model_save_path, mapping_fn):
    model = tf.keras.models.load_model(model_save_path, compile=False)
    model.summary()
    new_inputs = [model.inputs["q_term"], model.inputs["d_term"]]
    new_outputs = mapping_fn(model.output)
    new_model = tf.keras.models.Model(inputs=new_inputs, outputs=new_outputs)
    return new_model
