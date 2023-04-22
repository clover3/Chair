import json
import sys
import tensorflow as tf
from transformers import AutoTokenizer

from cpath import output_path
from misc_lib import path_join, TimeEstimator
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2, get_run_config_for_predict
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config

from trainer_v2.per_project.transparency.mmp.alignment.grad_extractor import GradExtractor
from trainer_v2.per_project.transparency.mmp.dev_analysis.alignment_prediction import compute_alignment
from trainer_v2.per_project.transparency.mmp.dev_analysis.when_term_frequency import enum_when_corpus
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy

# Extract local alignment

def main(args):
    run_config: RunConfig2 = get_run_config_for_predict(args)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    target_q_word = "when"
    target_q_word_id = tokenizer.vocab[target_q_word]

    def print_aligns(triplets):
        for j, token_id, score in triplets:
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            print("{} {} {}".format(j, token, score))

    def enum_qd_pairs():
        for query, doc_pos, doc_neg in enum_when_corpus():
            yield query, doc_pos
            yield query, doc_neg

    num_record = 13220
    ticker = TimeEstimator(num_record)
    strategy = get_strategy_from_config(run_config)
    out_f = open(path_join(output_path, "msmarco", "when_tf"), "w")
    print(strategy)
    tf.debugging.set_log_device_placement(True)
    print("run_config.common_run_config.batch_size", run_config.common_run_config.batch_size)
    with strategy.scope():
        extractor = GradExtractor(
            run_config.predict_config.model_save_path,
            run_config.common_run_config.batch_size,
            strategy
        )
        me_itr = extractor.encode(enum_qd_pairs())
        for me in me_itr:
            aligns = compute_alignment(me, target_q_word_id)
            # print(tokenizer.convert_ids_to_tokens(me.input_ids))
            logits = me.logits.tolist()
            # print(print_aligns(aligns))
            out_info = {'logits': logits, 'aligns': aligns}
            out_f.write(json.dumps(out_info) + "\n")
            ticker.tick()


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

