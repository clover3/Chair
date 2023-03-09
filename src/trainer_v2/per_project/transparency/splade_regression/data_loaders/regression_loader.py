import json
import sys
from collections import defaultdict

import tensorflow as tf
import os
from typing import List, Dict, Tuple, Iterable, Any

from transformers import AutoTokenizer

from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.per_project.transparency.splade_regression.data_loaders.iterate_data import iterate_triplet


def read_vector_json(file_path):
    f = open(file_path, "r")
    for line in f:
        j_obj = json.loads(line)
        yield j_obj


Triplet = Tuple[str, str, str]
def iterate_triplet_and_scores(text_path, vector_dir, max_n, line_per_job) \
        -> Iterable[Tuple[Triplet, Dict]]:
    text_itr = iterate_triplet(text_path)
    try:
        for i in range(max_n):
            file_name = f"{i}.jsonl"
            vector_path = os.path.join(vector_dir, file_name)
            c_log.info("Reading {} ".format(file_name))
            if not os.path.exists(vector_path):
                c_log.warning("{} does not exist ".format(file_name))
            else:
                j_iter = read_vector_json(vector_path)
                try:
                    for j in range(line_per_job):
                        j = next(j_iter)
                        triplet = next(text_itr)
                        yield triplet, j
                except StopIteration:
                    pass
    except StopIteration:
        pass


def iterate_triplet_and_scores_per_partition(
        partitioned_format_str, vector_dir, partition_no) \
        -> Iterable[Tuple[Triplet, Dict]]:
    text_path = partitioned_format_str.format(partition_no)
    text_itr = iterate_triplet(text_path)
    try:
        file_name = f"{partition_no}.jsonl"
        vector_path = os.path.join(vector_dir, file_name)
        c_log.info("Reading {} ".format(file_name))
        j_iter = read_vector_json(vector_path)
        for triplet, j in zip(text_itr, j_iter):
            yield triplet, j
    except StopIteration:
        pass


def apply_batch(itr: Iterable[Tuple], batch_size, drop_last=False) -> Iterable[Dict[int, List[Any]]]:
    batch: Dict[int, List[Any]] = defaultdict(list)
    for maybe_tuple in itr:
        for idx, item in enumerate(maybe_tuple):
            batch[idx].append(item)

        if len(batch[0]) == batch_size:
            yield batch
            batch = defaultdict(list)
    if not drop_last:
        if len(batch[0]):
            yield batch


XEncoded = Tuple[List[int], List[int]]


class VectorRegressionLoader:
    def __init__(
            self,
            text_path, vector_dir,
            max_partition=1000,
            line_per_job=10000,
            checkpoint_model_name="distilbert-base-uncased",
            vector_len=30522,
            max_length=512,
    ):
        self.text_path = text_path
        self.vector_dir = vector_dir
        self.max_partition = max_partition
        self.line_per_job = line_per_job
        self.vector_len = vector_len
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_model_name)

        self.max_length = max_length

    def iterate_text_j(self, partition=None) -> Iterable[Tuple[str, Tuple[List[List[int]], List[float]]]]:
        if partition is not None:
            pair_itr = iterate_triplet_and_scores_per_partition(self.text_path, self.vector_dir, partition)
        else:
            pair_itr = iterate_triplet_and_scores(
                self.text_path, self.vector_dir, self.max_partition, self.line_per_job)
        for triplet, j in pair_itr:
            query, d1, d2 = triplet
            text_d = {
                'q': query,
                'd_pos': d1,
                'd_neg': d2
            }
            for text_type in ["q", "d_pos", "d_neg"]:
                text = text_d[text_type]
                number_d = {}
                for role in ["indices", "values"]:
                    key = f"{text_type}_{role}"
                    number_d[role] = j[key]
                y = (number_d['indices'], number_d['values'])
                yield text, y

    def apply_tokenize(self, inputs) -> Tuple[XEncoded, Any]:
        text, y = inputs
        encoded = self.tokenizer(text)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        ret: Tuple = (input_ids, attention_mask), y
        return ret

    def build_sparse_tensor(self, inputs):
        x, y = inputs
        indices, values = y
        tensor = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[self.vector_len, ]
        )
        return x, tensor

    def iterate_tokenized(self, partition=None) -> Iterable[Tuple[XEncoded, Any]]:
        itr: Iterable[Tuple[str, Tuple[List[List[int]], List[float]]]] = self.iterate_text_j(partition)
        itr: Iterable[Tuple[XEncoded, Any]] = map(self.apply_tokenize, itr)
        return itr

    def iterate_batched_xy(self, partition=None, batch_size=None) -> Iterable[Dict[int, List]]:
        itr = self.iterate_text_j(partition)
        itr = map(self.apply_tokenize, itr)
        itr = map(self.build_sparse_tensor, itr)

        max_len = [self.max_length, self.max_length, self.vector_len]

        def pad_truncate(idx, items) -> List[tf.Tensor]:
            target_len = max_len[idx]
            truncated = [t[:target_len] for t in items]
            pad_len_list = [target_len - len(t) for t in truncated]
            padded_list = [tf.pad(item, [(0, pad_len)]) for item, pad_len in zip(truncated, pad_len_list)]
            return padded_list

        try:
            batch = defaultdict(list)
            for (i1, i2), sparse_vector in itr:
                dense_vector = tf.sparse.to_dense(sparse_vector)
                for idx, elem in enumerate([i1, i2, dense_vector]):
                    batch[idx].append(elem)

                if len(batch[0]) == self.batch_size:
                    batched = {}
                    for idx, items in batch.items():
                        batched[idx] = tf.stack(pad_truncate(idx, items), 0)
                    if len(batch[1]) == 0:
                        c_log.warning("len(batch[1]) == 0")
                    yield (batched[0], batched[1]), batched[2]
                    batch = defaultdict(list)

        except StopIteration:
            pass


def get_dev_dataloader():
    text_path = "C:\\work\\code\\chair\\data\\msmarco\\splade_triplets\\raw_head.tsv"
    vector_dir = "C:\work\code\chair\data\msmarco\splade_triplets\encoded"
    data_loader = VectorRegressionLoader(text_path, vector_dir, max_partition=1)
    return data_loader


def main():
    text_path = sys.argv[1]
    vector_dir = sys.argv[2]
    data_loader = VectorRegressionLoader(text_path, vector_dir)


if __name__ == "__main__":
    main()