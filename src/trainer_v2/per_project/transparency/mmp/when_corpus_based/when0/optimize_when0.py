import logging
import sys
from collections import defaultdict
from typing import List, Iterable
import numpy as np
from cpath import output_path
from misc_lib import path_join, batch_iter_from_entry_iter

from dataset_specific.msmarco.passage.passage_resource_loader import enum_grouped, enum_when_corpus_tokenized, FourItem
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import build_table
from trainer_v2.per_project.transparency.mmp.when_corpus_based.gradient_computer import \
    GoldPairBasedSamplerFromTokenized
import tensorflow as tf

from trainer_v2.per_project.transparency.voca_projector import BertVocaSpace, VocaSpaceIF, VocaSpace
from typing import List, Iterable, Callable, Dict, Tuple, Set


def main():
    try:
        run_name = sys.argv[1]
    except IndexError:
        run_name = "manual_grad"
    mapping: Dict[str, Dict[str, float]] = defaultdict(dict)
    mapping['when'] = build_table()
    c_log.setLevel(logging.DEBUG)
    itr: Iterable[FourItem] = enum_when0_tokenized()
    itr: Iterable[List[FourItem]] = enum_grouped(itr)
    batch_size = 100
    b_itr = batch_iter_from_entry_iter(itr, batch_size)
    sampler = GoldPairBasedSamplerFromTokenized(mapping)
    # voca_space: VocaSpaceIF = BertVocaSpace()
    c_log.info("Using VocaSpace")
    voca_space: VocaSpaceIF = VocaSpace(mapping['when'].keys())
    monitor_terms = ['when', '1940', 'happen', 'until', '29']
    def display_terms():
        d = {}
        for t in monitor_terms:
            token_id = voca_space.get_token_id(t)
            s = float(var_w[token_id].numpy())
            d[t] = "{0:5f}".format(s)
        c_log.info(str(d))

    def display_terms_grad():
        d = {}
        for t in monitor_terms:
            token_id = voca_space.get_token_id(t)
            s = float(grad_array[token_id])
            d[t] = "{0:5f}".format(s)
        c_log.info(str(d))

    def save_mapping(run_name, step):
        c_log.info("saving mapping")
        save_name = f"{run_name}_{step}"
        save_path = path_join(
            output_path, "msmarco", "passage", "when_trained_saved", save_name)
        f = open(save_path, "w")
        for k, v in mapping['when'].items():
            f.write(f"{k}\t{v}\n")

    def update_mapping(d):
        for k,v in d.items():
            mapping['when'][k] = v

    init_w = voca_space.dict_to_numpy(mapping['when'])
    var_w = tf.Variable(init_w, trainable=True)
    display_terms()
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for step, groups in enumerate(b_itr):
        c_log.info("Step %d: Computing for %d groups", step, len(groups))
        grad_dict, info = sampler.compute(groups)
        c_log.debug(info)
        grad_array: np.array = voca_space.dict_to_numpy(grad_dict)

        optimizer.apply_gradients([(grad_array, var_w)])
        cur_w = var_w.value().numpy()
        new_mapping = voca_space.numpy_to_dict(cur_w)
        update_mapping(new_mapping)
        display_terms()
        display_terms_grad()
    save_mapping(run_name, step)



if __name__ == "__main__":
    main()