import os
import random

import numpy as np

from cpath import output_path
from data_generator.NLI.nli import get_modified_data_loader
from data_generator.bert_input_splitter import get_sep_loc
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from misc_lib import Averager
from tf_v2_support import disable_eager_execution
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attention_opt import \
    TransformerAttentionOptModel
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attention_opt_wrap import \
    AttentionMaskOptimizer, init_model_for_inference, inference
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attnetion_opt_utils import AttnOptHP
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.segment_helper import \
    get_always_active_mask_w_input_ids


class NLIHP(AttnOptHP):
    num_classes = 3


def load_data(hp):
    vocab_filename = "bert_voca.txt"
    data_loader = get_modified_data_loader(get_tokenizer(), hp.seq_max, vocab_filename)
    return data_loader.get_dev_data()


def get_random_dropped_mask(x0, drop_rate):
    max_seq_length = len(x0)
    first_sep_loc, second_sep_loc = get_sep_loc(x0)
    l1 = first_sep_loc - 1
    l2 = (second_sep_loc - first_sep_loc) - 1
    total_l = l1 + l2 + 3

    def is_seg1(i):
        return 1 <= i < 1 + l1

    def is_seg2(i):
        return 2 + l1 <= i < 2 + l1 + l2

    n_drop = 0
    mask = np.ones([max_seq_length, max_seq_length])
    for i1 in range(max_seq_length):
        for i2 in range(max_seq_length):
            if (is_seg1(i1) and is_seg2(i2)) or (is_seg2(i1) and is_seg1(i2)):
                if random.random() < drop_rate:
                    mask[i1, i2] = 0
                    n_drop += 1
    return mask


def no_train_eval():
    disable_eager_execution()
    hp = NLIHP()
    save_path = os.path.join(output_path, "model", "runs", "nli512")
    item_list = load_data(hp)
    tokenizer = get_tokenizer()
    task, sess = init_model_for_inference(hp, save_path)
    print("Using random drop mask")
    for j in range(10):
        drop_rate = j / 10
        loss_1_averager = Averager()
        for item in item_list[:30]:
            x0, x1, x2, y = item
            x3 = get_always_active_mask_w_input_ids(x0)
            item_d = {
                'input_ids': x0,
                "input_mask": x1,
                "segment_ids": x2,
                "always_active_mask": x3
            }
            fetch_inf_mask = get_random_dropped_mask(x0, drop_rate)
            item_d["given_mask"] = fetch_inf_mask
            d = inference(sess, task, item_d)
            loss_1 = d['loss_1']
            loss_1_averager.append(loss_1)
        print("{}\t{}".format(drop_rate, loss_1_averager.get_average()))


def main():
    disable_eager_execution()
    hp = NLIHP()
    save_path = os.path.join(output_path, "model", "runs", "nli512")
    model = TransformerAttentionOptModel(hp)
    optimizer = AttentionMaskOptimizer(model, hp)
    optimizer.load_model(save_path)
    item_list = load_data(hp)
    num_steps = 200
    tokenizer = get_tokenizer()
    for item in item_list[1:]:
        print("-------------")
        x0, x1, x2, y = item
        print(pretty_tokens(tokenizer.convert_ids_to_tokens(x0), True))
        print(y)
        x3 = get_always_active_mask_w_input_ids(x0)
        item_d = {
            'input_ids': x0,
            "input_mask": x1,
            "segment_ids": x2,
            "always_active_mask": x3
        }
        optimizer.train(item_d, num_steps)
        fetch_inf_mask = optimizer.fetch_inf_mask(item_d)
        item_d["given_mask"] = fetch_inf_mask
        print("Getting infernece")
        task, sess = init_model_for_inference(hp, save_path)
        d = inference(sess, task, item_d)


if __name__ == "__main__":
    no_train_eval()