from collections import Counter, defaultdict

from cpath import at_output_dir
from data_generator.bert_input_splitter import get_first_seg
from data_generator.tokenizer_wo_tf import pretty_tokens
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.analysis.save_parsed import get_voca_list, convert_ids_to_tokens
from tlm.qtype.qtype_analysis import QTypeInstance


def parse_q_weight_output(raw_prediction_path):
    viewer = EstimatorPredictionViewer(raw_prediction_path)
    tokenizer = viewer.tokenizer
    voca_list = get_voca_list(tokenizer)

    counter = Counter()

    query_counter = defaultdict(Counter)
    # ticker = TimeEstimator(viewer.data_len)
    for e in viewer:
        # ticker.tick()
        qtype_weights_paired = e.get_vector("qtype_weights_paired")
        orig_logits_pair = e.get_vector("orig_logits_pair")
        drop_logits_pair = e.get_vector("drop_logits_pair")

        for sent_idx in [1, 2]:
            slide_idx = sent_idx - 1

            def get_q_d(key):
                input_ids = e.get_vector(key)
                # q, d = split_p_h_with_input_ids(input_ids, input_ids)
                q = get_first_seg(input_ids)
                # q_tokens = viewer.tokenizer.convert_ids_to_tokens(q)
                q_tokens = convert_ids_to_tokens(voca_list, q)
                # d_tokens = viewer.tokenizer.convert_ids_to_tokens(d)
                d_tokens = []
                return q_tokens, d_tokens

            orig_q, orig_d = get_q_d("input_ids{}".format(sent_idx))
            drop_q, drop_d = get_q_d("drop_input_ids{}".format(sent_idx))
            assert all([t1 == t2 for t1, t2 in zip(orig_d, drop_d)])
            qtype_weights = qtype_weights_paired[slide_idx]
            insts = QTypeInstance(orig_q, drop_q, orig_d, qtype_weights, int(sent_idx == 1))
            # orig_d = []

            q_pair_rep = "{}\t{}".format(pretty_tokens(orig_q, True), pretty_tokens(drop_q, True))

        def is_correct(logit_pair):
            return logit_pair[0] > logit_pair[1]

        correctness = is_correct(orig_logits_pair), is_correct(drop_logits_pair)
        # print("Orig: {0:.2f} > {1:.2f}".format(orig_logits_pair[0], orig_logits_pair[1]))
        # print("Drop: {0:.2f} > {1:.2f}".format(drop_logits_pair[0], drop_logits_pair[1]))
        query_counter[correctness][q_pair_rep] += 1
        counter[correctness] += 1



    counter_orig = Counter()
    counter_drop = Counter()

    for correctness, cnt in counter.items():
        print(correctness, cnt)
        for key, value in query_counter[correctness].items():
            print("{}\t{}".format(key, value))
        orig_correct, drop_correct = correctness
        counter_orig[orig_correct] += cnt
        counter_drop[drop_correct] += cnt

    def acc(counter):
        return counter[True] / sum(counter.values())

    print("orig", acc(counter_orig))
    print("drop", acc(counter_drop))


def main():
    run_name_list = ["mmd_qtype_J2_0", "mmd_qtype_I"]
    for run_name in run_name_list:
        print(run_name)
        raw_prediction_path = at_output_dir("qtype", run_name)
        parse_q_weight_output(raw_prediction_path)


if __name__ == "__main__":
    main()