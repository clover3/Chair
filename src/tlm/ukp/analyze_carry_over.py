import os
import pickle

import numpy as np

from cpath import output_path
from data_generator.tokenizer_wo_tf import pretty_tokens, get_tokenizer
from misc_lib import group_by, average


def analyze(lpp, info_path):
    lpp = pickle.load(open(lpp, "rb"))
    info = pickle.load(open(info_path, "rb"))

    info_d = {}
    for e in info:
        info_d[e['payload_id']] = e

    tokenizer = get_tokenizer()
    def get_segment_similarity(input_ids):
        first_sep_loc, second_sep_loc = get_sep(input_ids)

        if first_sep_loc is not None and second_sep_loc is not None:
            first = input_ids[1:first_sep_loc]
            second = input_ids[first_sep_loc+1:second_sep_loc]

            cnt = 0
            second_set = set(second)
            for token_id in first:
                if token_id in second_set:
                    cnt += 1

            return cnt
        else:
            return -1

    def get_sep(input_ids):
        SEP_ID = 102
        first_sep_loc = None
        for loc, token_id in enumerate(input_ids):
            if token_id == SEP_ID:
                first_sep_loc = loc
                break
        second_sep_loc = None
        for idx2 in range(first_sep_loc + 1, len(input_ids)):
            if input_ids[idx2] == SEP_ID:
                second_sep_loc = idx2
        return first_sep_loc, second_sep_loc

    def get_two_text(input_ids):
        first_sep_loc, second_sep_loc = get_sep(input_ids)

        if first_sep_loc is not None and second_sep_loc is not None:
            first = input_ids[1:first_sep_loc]
            second = input_ids[first_sep_loc+1:second_sep_loc]
            first = tokenizer.convert_ids_to_tokens(first)
            second = tokenizer.convert_ids_to_tokens(second)
            return pretty_tokens(first), pretty_tokens(second)

        else:
            return "",""

    summary = []
    for batch in lpp:
        input_ids = batch["input_ids"]
        batch_size, seq_length = input_ids.shape
        losses = batch["masked_lm_example_loss"]
        losses = np.reshape(losses, [batch_size, -1])
        instance_id_list = batch["instance_id"]

        for i in range(batch_size):
            d = info_d[instance_id_list[i][0]]


            summary.append((d, losses[i], input_ids[i]))


    def tuple_to_key(tuple):
        d, _ , _ = tuple
        return d['doc_id'], d['st'], d['ed']

    for key, entries in group_by(summary, tuple_to_key).items():
        doc_id, st, ed = key
        print("doc_id={}  ({}~{})".format(doc_id, st, ed))

        sim_list = []
        loss_diff_list = []
        for d, losses, input_ids in entries:
            if d['hint_loc'] == -1:
                print(first)
                base = cur_score

            if d['hint_loc'] is not -1:
                sim_list.append(get_segment_similarity(input_ids))
                diff = cur_score - base
                loss_diff_list.append(diff)

        avg_sim = average(sim_list)
        avg_diff = average(loss_diff_list)
        base_rate = avg_diff / avg_sim
        for d, losses, input_ids in entries:
            first, second = get_two_text(input_ids)
            cur_score = np.sum(losses)
            if d['hint_loc']  == -1:
                print(first)
                base = cur_score
            diff = cur_score - base
            sim_diff = get_segment_similarity(input_ids)-avg_sim

            avg_sim
            print(d['hint_loc'], cur_score - base, sim_diff, second)












if __name__ == "__main__":
    lpp = os.path.join(output_path, "carry_over.pickle")
    info_path = os.path.join(output_path, "abortion.info")
    analyze(lpp, info_path)