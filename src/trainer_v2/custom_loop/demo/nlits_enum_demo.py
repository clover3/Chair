import sys
from collections import Counter

import numpy as np
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from misc_lib import two_digit_float
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
from trainer_v2.custom_loop.per_task.nli_ts_util import enum_hypo_token_tuple_from_tokens
from trainer_v2.custom_loop.run_config2 import get_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    # load model
    tokenizer = get_tokenizer()

    def get_inputs():
        # sent1 = input("Premise: ")
        # sent2 = input("Hypo: ")
        tab_sep = input("Premise[Tab]Hypo: ")
        sent1, sent2 = tab_sep.split("\t")
        return sent1.split(), sent2.split()

    run_config = get_run_config2(args)
    run_config.print_info()
    nlits = get_local_decision_nlits_core(run_config, "concat")
    def sb_tokenize(tokens):
        output = []
        for t in tokens:
            output.extend(tokenizer.tokenize(t))
        return output

    window_size = 1
    while True:
        #   Receive Input
        p_tokens, h_tokens = get_inputs()
        #   Enum partition choices
        print("prem:", " ".join(p_tokens))
        print("hypo:", " ".join(h_tokens))
        p_sb_tokens = sb_tokenize(p_tokens)

        payload = []
        x_list = []
        for h_first, h_second, st, ed in enum_hypo_token_tuple_from_tokens(tokenizer, h_tokens, window_size):
            x = nlits.encode_fn(p_sb_tokens, h_first, h_second)
            x_list.append(x)
            payload.append((h_first, h_second))

        probs = nlits.predict(x_list)
        l_decisions, g_decisions = probs
        print("{} items".format(len(payload)))
        decisions_summary = []
        for idx in range(len(payload)):
            l_first, l_second = l_decisions[idx]
            g_decision = g_decisions[0][idx]

            g_pred = np.argmax(g_decision)
            l1_pred = np.argmax(l_first)
            l2_pred = np.argmax(l_second)
            d = {'g': g_pred,
                 'g_decision': g_decision,
                 'l1': l1_pred,
                 'l_first': l_first,
                 'l_second': l_second,
                 'l2': l2_pred}
            decisions_summary.append(d)

        g_list = list(map(lambda d: d['g'], decisions_summary))
        l1_list = map(lambda d: d['l1'], decisions_summary)
        l2_list = map(lambda d: d['l2'], decisions_summary)
        g_summary = Counter(g_list)
        print(list(g_list))

        major, n = list(g_summary.most_common())[0]
        print("Major decision: {}".format(major))

        for idx, g in enumerate(g_list):
            if g != major:
                h_first, h_second = payload[idx]
                g_decision = decisions_summary[idx]['g_decision']
                l_first = decisions_summary[idx]['l_first']
                l_second = decisions_summary[idx]['l_second']
                def probs_to_str(probs):
                    s = ", ".join(map(two_digit_float, probs))
                    return "[ " + s + " ]"

                print("  {} -> Combined".format(probs_to_str(g_decision)))
                print("  {1} {0}".format(pretty_tokens(h_first, True), probs_to_str(l_first)))
                print("  {1} {0}".format(pretty_tokens(h_second, True), probs_to_str(l_second)))
                print("")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
