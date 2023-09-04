import json
import sys
from transformers import AutoTokenizer

from cpath import output_path
from misc_lib import path_join
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config_for_predict
from trainer_v2.per_project.transparency.mmp.alignment.network.alignment_predictor import AlignmentPredictor, \
    compute_alignment_first_layer

from trainer_v2.per_project.transparency.mmp.alignment.grad_extractor import ModelEncoded
from trainer_v2.per_project.transparency.mmp.dev_analysis.when_term_frequency import enum_when_corpus
from trainer_v2.train_util.arg_flags import flags_parser
from typing import Callable, Any


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

