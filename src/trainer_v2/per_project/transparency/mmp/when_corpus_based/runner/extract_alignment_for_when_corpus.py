import json
import sys
import tensorflow as tf
from transformers import AutoTokenizer

from cpath import output_path
from misc_lib import path_join, TimeEstimator
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config_for_predict
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.mmp.alignment.alignment_predictor import AlignmentPredictor, \
    compute_alignment_first_layer

from trainer_v2.per_project.transparency.mmp.alignment.grad_extractor import GradExtractor, ModelEncoded
from trainer_v2.per_project.transparency.mmp.dev_analysis.when_term_frequency import enum_when_corpus
from trainer_v2.train_util.arg_flags import flags_parser
from typing import Iterable, Callable, Any


def extract_save_align(
        compute_alignment_fn: Callable[[ModelEncoded], Any],
        qd_itr, run_config, save_path, num_record):
    ticker = TimeEstimator(num_record)
    strategy = get_strategy_from_config(run_config)
    out_f = open(save_path, "w")
    c_log.info("{}".format(strategy))
    tf.debugging.set_log_device_placement(True)
    with strategy.scope():
        extractor = GradExtractor(
            run_config.predict_config.model_save_path,
            run_config.common_run_config.batch_size,
            strategy
        )
        me_itr: Iterable[ModelEncoded] = extractor.encode(qd_itr)
        for me in me_itr:
            aligns = compute_alignment_fn(me)
            logits = me.logits.tolist()
            out_info = {'logits': logits, 'aligns': aligns}
            out_f.write(json.dumps(out_info) + "\n")
            ticker.tick()


# def main(args):
#     run_config: RunConfig2 = get_run_config_for_predict(args)
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     target_q_word = "when"
#     target_q_word_id = tokenizer.vocab[target_q_word]
#     compute_alignment_fn = compute_alignment
#     save_path = path_join(output_path, "msmarco", "when_tf")
#     extract_save_aligns(compute_alignment_fn, run_config, target_q_word_id, save_path)


# Extract local alignment

def extract_save_aligns_when(
        compute_alignment_fn: Callable[[ModelEncoded], Any],
        run_config,
        save_path):
    def enum_qd_pairs():
        for query, doc_pos, doc_neg in enum_when_corpus():
            yield query, doc_pos
            yield query, doc_neg
    num_record = 13220
    qd_itr = enum_qd_pairs()

    align_predictor = AlignmentPredictor(run_config, compute_alignment_fn)
    out_itr = align_predictor.predict_for_qd_iter(qd_itr, num_record)
    out_f = open(save_path, "w")
    for out_info in out_itr:
        out_f.write(json.dumps(out_info) + "\n")


def main(args):
    run_config: RunConfig2 = get_run_config_for_predict(args)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    target_q_word = "when"
    target_q_word_id = tokenizer.vocab[target_q_word]
    print("Using first layer's attention to extract alignment")
    def compute_alignment_fn(me: ModelEncoded):
        return compute_alignment_first_layer(me, target_q_word_id)
    save_path = path_join(output_path, "msmarco", "when_tf_l1.jsonl")
    extract_save_aligns_when(compute_alignment_fn, run_config, save_path)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

