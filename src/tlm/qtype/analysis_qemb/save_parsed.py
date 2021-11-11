from typing import List

from cache import save_to_pickle
from cpath import at_output_dir
from data_generator.bert_input_splitter import get_first_seg
from misc_lib import TimeEstimator
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.qtype_analysis import QTypeInstance


def get_voca_list(tokenizer):
    voca_d = tokenizer.inv_vocab
    voca_len = max(voca_d.keys()) + 1

    voca = ['' for _ in range(voca_len)]
    for k, v in voca_d.items():
        voca[k] = v
    return voca


def convert_ids_to_tokens(voca_list, ids):
    output = []
    for item in ids:
        output.append(voca_list[item])
    return output


def parse_q_weight_output(raw_prediction_path) -> List[QTypeInstance]:
    all_insts = []
    viewer = EstimatorPredictionViewer(raw_prediction_path)
    tokenizer = viewer.tokenizer
    voca_list = get_voca_list(tokenizer)

    ticker = TimeEstimator(viewer.data_len)
    for e in viewer:
        ticker.tick()
        qtype_weights_paired = e.get_vector("qtype_weights_paired")
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
            # orig_d = []
            insts = QTypeInstance(orig_q, drop_q, orig_d, qtype_weights, int(sent_idx == 1))
            all_insts.append(insts)
    return all_insts


def save_parsed():
    raw_prediction_path = at_output_dir("qtype", "mmd_qtype_G_pred")
    all_insts = parse_q_weight_output(raw_prediction_path)
    save_to_pickle(all_insts, "mmd_qtype_G_pred.parsed")


if __name__ == "__main__":
    save_parsed()
