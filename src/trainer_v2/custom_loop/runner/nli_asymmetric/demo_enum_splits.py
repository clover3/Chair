import os
import sys
from typing import List

import numpy as np

from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from data_generator2.segmented_enc.seg_encoder_common import encode_two_segments
from dataset_specific.mnli.mnli_reader import NLIPairData
from list_lib import MaxKeyValue
from misc_lib import two_digit_float
from tlm.data_gen.base import get_basic_input_feature_as_list
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig200_200
from trainer_v2.custom_loop.per_task.nli_ts_util import batch_shaping, load_local_decision_nli

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser


def iter_alamri():
    problems: List[AlamriProblem] = load_alamri_problem()

    for p in problems:
        yield NLIPairData(p.text1, p.text2, "neutral", "")



@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)
    model_config = ModelConfig200_200()
    model_path = run_config.eval_config.model_save_path
    predictor = load_local_decision_nli(model_path)
    tokenizer = get_tokenizer()
    def encode_prem(prem_text):
        tokens = tokenizer.tokenize(prem_text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, model_config.max_seq_length1,
                                                                             tokens, segment_ids)
        return input_ids, segment_ids

    segment_len = int(model_config.max_seq_length2 / 2)
    window_size = 3

    def enum_hypo_tuples(hypo_text, window_size):
        space_tokenized_tokens = hypo_text.split()
        st = 0
        def sb_tokenize(tokens):
            output = []
            for t in tokens:
                output.extend(tokenizer.tokenize(t))
            return output

        while st < len(space_tokenized_tokens):
            ed = st + window_size
            first_a = space_tokenized_tokens[:st]
            second = space_tokenized_tokens[st:ed]
            first_b = space_tokenized_tokens[ed:]

            first = sb_tokenize(first_a) + ["[MASK]"] + sb_tokenize(first_b)
            second = sb_tokenize(second)

            all_input_ids, all_input_mask, all_segment_ids = encode_two_segments(tokenizer, segment_len, first, second)
            yield all_input_ids, all_segment_ids
            st += window_size

    # reader = MNLIReader()
    # itr = reader.load_split("dev")
    itr = iter_alamri()
    for e in itr:
        print("prem: ", e.premise)
        prem_inputs = encode_prem(e.premise)
        p_x0, p_x1 = prem_inputs

        most_neutral = MaxKeyValue()
        most_contradiction = MaxKeyValue()

        for hypo_inputs in enum_hypo_tuples(e.hypothesis, window_size):
            h_x0, h_x1 = hypo_inputs
            x = p_x0, p_x1, h_x0, h_x1
            x = tuple(map(batch_shaping, x))
            l_decision, g_decision = predictor(x)
            g_decision = g_decision[0]
            l_decision = l_decision[0]
            input_ids1, _, input_ids2, _ = x
            def format_prob(probs):
                return ", ".join(map(two_digit_float, probs))

            # g_decision_s = format_prob(g_decision)
            g_pred = np.argmax(g_decision)
            l_pred = np.argmax(l_decision, axis=1)
            print(" Pred: {} ({})".format(g_pred, g_decision),  " label :", e.get_label_as_int())
            h_tokens = tokenizer.convert_ids_to_tokens(h_x0)

            h_first = h_tokens[:100]
            h_second = h_tokens[100:]
            print(" hypo1 ({}): {}".format(format_prob(l_decision[0]), pretty_tokens(h_first, True)))
            print(" hypo2 ({}): {}".format(format_prob(l_decision[1]), pretty_tokens(h_second, True)))

            most_neutral.update(h_second, l_decision[1][1])
            most_contradiction.update(h_second, l_decision[1][2])
        print("most_neutral:", pretty_tokens(most_neutral.max_key), most_neutral.max_value)
        print("most_contradiction:", pretty_tokens(most_contradiction.max_key), most_contradiction.max_value)
        input("Press enter to continue")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
