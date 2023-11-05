import sys

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from iter_util import load_jsonl
from misc_lib import TimeEstimator
from trainer_v2.custom_loop.definitions import HFModelConfigType
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel
from trainer_v2.custom_loop.per_task.ts_util import get_local_decision_layer_from_model_by_shape
from trainer_v2.per_project.transparency.mmp.pep.demo_util import PEPLocalDecision
from trainer_v2.per_project.transparency.mmp.pep.term_pair_scoring_runner.read_segment_log import load_segment_log
from trainer_v2.train_util.get_tpu_strategy import get_strategy


class ModelConfig16_1(HFModelConfigType):
    max_seq_length = 16
    num_classes = 1
    model_type = "bert-base-uncased"


def load_model(
        new_model_config: HFModelConfigType, model_save_path):
    task_model = TwoSegConcatLogitCombineTwoModel(
        new_model_config, CombineByScoreAdd)
    task_model.build_model(None)

    model = task_model.point_model

    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(model_save_path).expect_partial()
    local_decision_layer = get_local_decision_layer_from_model_by_shape(
        model, new_model_config.num_classes)
    new_outputs = [local_decision_layer.output, model.outputs]
    new_model = tf.keras.models.Model(inputs=model.input, outputs=new_outputs)
    return new_model


def predict_with_small_window_model(
        model_path,
        log_path,
        score_d,
        n=2000):
    strategy = get_strategy()
    with strategy.scope():
        model_config = ModelConfig16_1()
        model = load_model(model_config, model_path)
        pep = PEPLocalDecision(model_config, model_path=None, model=model)

    tokenizer = get_tokenizer()

    ticker = TimeEstimator(n)
    payload = []
    info = []
    for q_term, d_term in score_d:
        q_tokens = tokenizer.tokenize(q_term)
        d_tokens = tokenizer.tokenize(d_term)

        q_tokens = ["[MASK]"] + q_tokens + ["[MASK]"]
        d_tokens = ["[MASK]"] * 4 + d_tokens + ["[MASK]"] * 4

        info.append((q_term, d_term))
        payload.append((q_tokens, d_tokens))
        if len(payload) >= n:
            break
        ticker.tick()

    scores = pep.score_fn(payload)

    out_f = open(log_path, "w")
    for (q_term, d_term), score in zip(info, scores):
        out_f.write(f"{q_term}\t{d_term}\t{score}\n")


def main():
    # Goal: check which span pairs are important in PEP.
    #   Method: For each qd, apply a few pairs, record top scoring one.
    #
    #   How to handle exact match?
    #       - Exact match having high score is trivial. So it may not be worth checking it.
    #       - However, they don't always result in the same score.
    #            Some (rare words) has higher score than other (frequent ones)
    #
    jsonl = load_jsonl(sys.argv[1])
    score_d = load_segment_log(jsonl)

    model_path = sys.argv[2]
    log_path = sys.argv[3]
    predict_with_small_window_model(model_path, log_path, score_d)


if __name__ == "__main__":
    main()
