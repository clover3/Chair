import sys

from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
from trainer_v2.custom_loop.run_config2 import get_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    run_config = get_run_config2(args)
    run_config.print_info()
    nlits = get_local_decision_nlits_core(run_config, "concat")
    tokenizer = get_tokenizer()
    while True:
        sent1 = input("Premise: ")
        sent2 = input("Partial Hypo: ")
        p_tokens = tokenizer.tokenize(sent1)
        h_first = []
        h_second = tokenizer.tokenize(sent2)
        x = nlits.encode_fn(p_tokens, h_first, h_second)
        probs = nlits.predict([x])
        l_decisions, g_decisions = probs
        print(l_decisions.shape)
        g_decision = g_decisions[0][0]
        l_decision = l_decisions[0]
        print((sent1, sent2))
        print("g_decision: ", g_decision)
        print("l_decision: ", l_decision)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
