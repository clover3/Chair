
import os
import sys

import tensorflow as tf
from transformers import AutoTokenizer

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.alignment.network.first_hidden_pairwise import GAlignFirstHiddenTwoModel
from trainer_v2.per_project.transparency.mmp.alignment.predict_table.old.predict_d_terms_mmp_train import predict_d_terms_per_job_and_save
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from cpath import output_path
from misc_lib import path_join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return GAlignFirstHiddenTwoModel(tokenizer)


def fetch_align_probe_fn(outputs):
    # GAlign4
    scores = outputs['align_probe']["align_pred"][:, 0]
    return scores


def load_model(model_save_path):
    network = build_model()
    network.load_checkpoint(model_save_path)

    model = network.get_inference_model()
    model.summary()
    input_d = {item.name: item for item in model.input}
    new_inputs = [input_d["q_term"], input_d["d_term"]]
    new_outputs = fetch_align_probe_fn(model.output)
    new_model = tf.keras.models.Model(inputs=new_inputs, outputs=new_outputs)
    return new_model


def main():
    strategy = get_strategy()
    with strategy.scope():
        c_log.info(__file__)
        model_save_path = sys.argv[1]
        job_no = int(sys.argv[2])
        model_name = sys.argv[3]
        save_dir_path = path_join(output_path, "msmarco", "passage",
                                  f"candidate_building_{model_name}")

        def get_model():
            return load_model(model_save_path)

        predict_d_terms_per_job_and_save(get_model, job_no, save_dir_path)


if __name__ == "__main__":
    main()