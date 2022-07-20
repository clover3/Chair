import itertools
import os
import random

import numpy as np
from numpy.linalg import norm

from arg.qck.encode_common import encode_single
from cpath import at_output_dir, get_bert_config_path, common_model_dir_root
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.genex.cls_getter import CLSPooler
from list_lib import right
from misc_lib import get_first, group_by, Averager, TimeEstimator
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config


class CLSGetter:
    def __init__(self):
        self.cls_pooler = CLSPooler()
        bert_params = load_bert_config(get_bert_config_path())
        self.cls_pooler.build_model(bert_params)
        model_path = os.path.join(common_model_dir_root, "runs", "uncased_L-12_H-768_A-12", "bert_model.ckpt")
        self.cls_pooler.init_checkpoint(model_path)
        self.tokenizer = get_tokenizer()
        self.max_seq_length = 512
    def encode(self, text):
        tokens = self.tokenizer.tokenize(text)
        input_ids, input_mask, segment_ids = encode_single(self.tokenizer, tokens, self.max_seq_length)
        return input_ids

    def get(self, text):
        input_ids = self.encode(text)
        input_ids = np.expand_dims(input_ids, 0)
        batch_arr = self.cls_pooler.model.predict(input_ids)
        ret = batch_arr[0]
        assert len(ret) == 768
        return ret

    def get_from_list(self, text_list):
        input_ids = np.stack(list(map(self.encode, text_list)), 0)
        return self.cls_pooler.model.predict(input_ids)


def load_data(name):
    file_path = at_output_dir("Youngwoo-TOIS-R2", name)
    return open(file_path, "r").readlines()


def calc_sim():
    queries = load_data("q.txt")
    texts = load_data("t.txt")

    cls_getter = CLSGetter()

    data = list(zip(queries, texts))
    grouped = group_by(data, get_first)
    n_group = len(grouped)
    all_combs = list(itertools.combinations(range(n_group), 2))
    random.shuffle(all_combs)
    sample_combs = all_combs[:50]
    print("Total of {} combs. select 50".format(len(all_combs)))

    avg_over = Averager()
    np_vector_list_list = []
    ticker = TimeEstimator(len(grouped))
    for query, entries in grouped.items():
        ticker.tick()
        text_list = right(entries)
        raw_np_vector = cls_getter.get_from_list(text_list)
        np_vector = raw_np_vector / norm(raw_np_vector, axis=1, keepdims=True)
        np_vector_list_list.append(np_vector)
        sims = np.dot(np_vector, np.transpose(np_vector, [1, 0]))
        n = len(np_vector)
        diag_sum = sum(sims.diagonal())
        n_diag = len(sims)
        avg = (np.sum(sims) - diag_sum) / (n*n - n_diag)
        avg_over.append(avg)

    print("avg in same topic", avg_over.get_average())
    avg_diff = Averager()
    for g1, g2 in sample_combs:
        v1 = np_vector_list_list[g1]
        v2 = np_vector_list_list[g2]
        sims = np.dot(v1, np.transpose(v2, [1, 0]))

        avg_diff.append(np.mean(sims))

    print("avg in diff topic", avg_diff.get_average())





def main():
    getter = CLSGetter()
    ret = getter.get("Text")


if __name__ == "__main__":
    calc_sim()

