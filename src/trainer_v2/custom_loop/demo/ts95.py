import sys

from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_nli_prediction_summary_str
from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    run_config = get_eval_run_config2(args)
    run_config.print_info()
    nlits = get_local_decision_nlits_core(run_config, "concat_wmask")
    tokenizer = get_tokenizer()
    while True:
        sent1 = input("Premise: ")
        sent2 = input("Hypo1[Hypo2]hypo1: ")

        p_tokens = tokenizer.tokenize(sent1)
        h_tokens_w_mark = tokenizer.tokenize(sent2)
        st = h_tokens_w_mark.index("[")
        ed = h_tokens_w_mark.index("]")
        # A / B / C
        # A = st
        # C =
        h_tokens_1_a = h_tokens_w_mark[:st]
        h_tokens_2 = h_tokens_w_mark[st+1:ed]
        h_tokens_1_b = h_tokens_w_mark[ed+1:]
        h_tokens_new = h_tokens_1_a + h_tokens_2 + h_tokens_1_b
        a_seg_len = st
        b_seg_len = ed-st-1
        c_seg_len = len(h_tokens_new) - a_seg_len - b_seg_len
        mask = [0] * a_seg_len + [1] * b_seg_len + [0] * c_seg_len
        x = nlits.encode_fn(p_tokens, h_tokens_new, mask)
        probs = nlits.predict([x])
        l_decisions, g_decisions = probs
        print(l_decisions.shape)
        g_decision = g_decisions[0][0]
        l_decision = l_decisions[0]
        l_first, l_second = l_decision
        print(sent1)
        print(h_tokens_1_a, h_tokens_1_b)
        print(h_tokens_2)
        print("g_decision: ", make_nli_prediction_summary_str(g_decision))
        print("l_decision1: ", make_nli_prediction_summary_str(l_first))
        print("l_decision2: ", make_nli_prediction_summary_str(l_second))


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
