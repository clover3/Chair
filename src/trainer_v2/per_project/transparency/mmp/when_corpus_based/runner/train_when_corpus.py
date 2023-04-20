import itertools
from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np
from transformers import AutoTokenizer
from cpath import output_path
from misc_lib import path_join, TELI

from dataset_specific.msmarco.passage.passage_resource_loader import enum_all_when_corpus, enum_grouped, FourStr
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import build_table
from trainer_v2.per_project.transparency.mmp.when_corpus_based.gradient_computer import GoldPairBasedSampler
import tensorflow as tf

class BertVocaSpace:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.inv_vocab = {v:k for k, v in self.tokenizer.vocab.items()}
        self.voca_size = 1 + max(self.tokenizer.vocab.values())

    def dict_to_numpy(self, number: Dict[str, float]) -> np.array:
        arr = np.zeros([self.voca_size])
        for word, score in number.items():
            for subword in self.tokenizer.tokenize(word):
                token_id = self.tokenizer.vocab[subword]
                arr[token_id] += score
                break
        return arr

    def numpy_to_dict(self, arr: np.array) -> np.array:
        out_d = {}
        non_zero_indices, = arr.nonzero()
        for token_id in non_zero_indices:
            token = self.inv_vocab[token_id]
            out_d[token] = arr[token_id]
        return out_d


def main():
    mapping = defaultdict(dict)
    mapping['when'] = build_table()

    itr: Iterable[FourStr] = enum_all_when_corpus()
    itr: Iterable[List[FourStr]] = enum_grouped(itr)
    sampler = GoldPairBasedSampler(mapping)
    voca_space = BertVocaSpace()
    monitor_terms = ['when', '1940', 'happen', 'until', '29']

    def display_terms():
        d ={}
        for t in monitor_terms:
            token_id = voca_space.tokenizer.vocab[t]
            s = float(var_w[token_id].numpy())
            d[t] = s
        c_log.info(str(d))

    def save_mapping():
        c_log.info("saving mapping")
        save_path = path_join(
            output_path, "msmarco", "passage", "when_trained")
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

    for groups in group_iter():
        c_log.info("Computing for {} groups".format(len(groups)))
        grad_dict = sampler.compute(groups)
        grad_array: np.array = voca_space.dict_to_numpy(grad_dict)
        optimizer.apply_gradients([(grad_array, var_w)])

        cur_w = var_w.value().numpy()
        new_mapping = voca_space.numpy_to_dict(cur_w)
        update_mapping(new_mapping)
        display_terms()
        save_mapping()


if __name__ == "__main__":
    main()