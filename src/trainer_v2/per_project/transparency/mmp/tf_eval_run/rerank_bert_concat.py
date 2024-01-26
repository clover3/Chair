import os.path
import sys
from typing import List

import tensorflow as tf
from omegaconf import OmegaConf

from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.neural_network_def.cat_pair import BERTConcatModel
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf2
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_qd_encoder
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def load_bert_concat_model(model_path, batch_size=16, max_seq_length=256):
    c_log.info("Loading model from %s", model_path)

    model_config = ModelConfig512_1()
    task_model = BERTConcatModel(model_config)
    task_model.build_model(None)
    inference_model = task_model.point_model
    checkpoint = tf.train.Checkpoint(inference_model)
    checkpoint.restore(model_path).expect_partial()

    new_model = tf.keras.models.Model(
        inputs=inference_model.inputs, outputs=inference_model.output[0])

    qd_encoder = get_qd_encoder(max_seq_length)

    def score_fn(qd_list: List):
        dataset = qd_encoder(qd_list)
        dataset = dataset.batch(batch_size)
        output = inference_model.predict(dataset)
        return output

    return score_fn


def main():
    dataset_conf_path = sys.argv[1]
    model_path = sys.argv[2]
    model_step = os.path.basename(model_path)
    model_name = os.path.basename(os.path.dirname(model_path))
    _model, step = model_step.split("_")
    run_name = f"{model_name}_{step}"

    c_log.info("Building scorer")
    max_seq_len = 512

    strategy = get_strategy()
    conf = OmegaConf.create({
        'run_name': run_name,
        'dataset_conf_path': dataset_conf_path,
        'outer_batch_size': 256,
    })

    with JobContext(f"{run_name}_eval"):
        with strategy.scope():
            score_fn = load_bert_concat_model(model_path, max_seq_length=max_seq_len)
            run_rerank_with_conf2(score_fn, conf)


if __name__ == "__main__":
    main()
