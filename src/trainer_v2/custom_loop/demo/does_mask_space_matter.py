import random
import sys
from collections import Counter

import numpy as np

from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_nli_prediction_summary_str
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.mnli.mnli_reader import MNLIReader
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
from trainer_v2.custom_loop.run_config2 import get_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    run_config = get_run_config2(args)
    run_config.print_info()
    nlits = get_local_decision_nlits_core(run_config, "concat_wmask")
    reader = MNLIReader()
    itr = iter(reader.get_dev())

    tokenizer = get_tokenizer()
    while True:
        nli_pair = next(itr)
        print("Premise: ", nli_pair.premise)
        print("Hypothesis: ", nli_pair.hypothesis)
        p_tokens = tokenizer.tokenize(nli_pair.premise)
        h_tokens = tokenizer.tokenize(nli_pair.hypothesis)

        ws = 1
        for st in range(len(h_tokens)):
            ed = st + ws
            if ed > len(h_tokens):
                break
            h_tokens_2 = h_tokens[st:ed]

            n_before = st
            n_after = len(h_tokens) - ed

            pad_info = []
            pad_info.append((n_before, n_after))

            for _ in range(10):
                n_before_other = random.randint(0, n_before + 5)
                n_after_other = random.randint(0, n_after + 5)
                pad_info.append((n_before_other, n_after_other))

            x_list = []
            for n_before, n_after in pad_info:
                h_tokens_new = ["[PAD]"] * n_before + h_tokens_2 + ["[PAD]"] * n_after
                mask = [0] * n_before + [1] * len(h_tokens_2) + [0] * n_after
                x = nlits.encode_fn(p_tokens, h_tokens_new, mask)
                x_list.append(x)

            probs = nlits.predict(x_list)
            l_decisions, g_decisions = probs
            predictions = []
            for i in range(len(pad_info)):
                l_decision = l_decisions[i]
                l_first, l_second = l_decision
                pred = np.argmax(l_second)
                predictions.append(pred)

            g_summary = Counter(predictions)
            major, n = list(g_summary.most_common())[0]
            all_equal = len(g_summary.keys()) == 1
            if all_equal:
                continue

            print(f"{h_tokens_2}  label={major}  n_before,n_after={pad_info[0]}")
            for i in range(len(pad_info)):
                if predictions[i] != major:
                    n_before, n_after = pad_info[i]
                    print(f" label={predictions[i]}  n_before,n_after={n_before, n_after}")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
