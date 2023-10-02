import sys
from collections import defaultdict, Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np
from transformers import AutoTokenizer
from cpath import output_path
from misc_lib import path_join, TELI

from dataset_specific.msmarco.passage.passage_resource_loader import enum_grouped, FourStr
from dataset_specific.msmarco.passage.processed_resource_loader import enum_all_when_corpus
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import build_table_when_avg
from trainer_v2.per_project.transparency.mmp.when_corpus_based.gradient_computer import GoldPairBasedSampler
import tensorflow as tf

from trainer_v2.per_project.transparency.voca_projector import BertVocaSpace


def main():
    try:
        run_name = sys.argv[1]
    except IndexError:
        run_name = "manual_grad"
    mapping = defaultdict(dict)
    mapping['when'] = build_table_when_avg()

    itr: Iterable[FourStr] = enum_all_when_corpus()
    itr: Iterable[List[FourStr]] = enum_grouped(itr)
    sampler = GoldPairBasedSampler(mapping)
    voca_space = BertVocaSpace()
    monitor_terms = ['when', '1940', 'happen', 'until', '29']

    def display_terms():
        d ={}
        for t in monitor_terms:
            token_id = voca_space.get_token_id(t)
            s = float(var_w[token_id].numpy())
            d[t] = s
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
    batch_size = 100

    def group_iter() -> Iterable[List[List[FourStr]]]:
        batch = []
        for t in itr:
            batch.append(t)
            if len(batch) == batch_size:
                yield batch
                batch = []
        yield batch

    for step, groups in enumerate(group_iter()):
        c_log.info("Computing for {} groups".format(len(groups)))
        grad_dict = sampler.compute(groups)
        grad_array: np.array = voca_space.dict_to_numpy(grad_dict)
        optimizer.apply_gradients([(grad_array, var_w)])
        print(sampler.time_d)
        cur_w = var_w.value().numpy()
        new_mapping = voca_space.numpy_to_dict(cur_w)
        update_mapping(new_mapping)
        display_terms()
        save_mapping(run_name, step)

        sampler.time_d = Counter()


if __name__ == "__main__":
    main()