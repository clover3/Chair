from collections import defaultdict
import tensorflow as tf
from typing import List, Iterable, Callable, Dict, Tuple, Set
from tensorflow.python.distribute.distribute_lib import Strategy
from transformers import AutoTokenizer

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.train_loop_helper import fetch_metric_result
from trainer_v2.per_project.transparency.splade_regression.data_loaders.iterate_data import iterate_triplet
from trainer_v2.per_project.transparency.splade_regression.path_helper import partitioned_triplet_path_format_str

pairwise_roles = ["q", "d1", "d2"]


def load_pairwise_eval_data() -> List[Tuple[str, str, str]]:
    c_log.info("load_pairwise_eval_data")
    # target_partition = list(range(1000, 1010))
    target_partition = list(range(1000, 1001))
    partitioned_format_str = partitioned_triplet_path_format_str()
    triplet_list = []
    for i in target_partition:
        text_path = partitioned_format_str.format(i)
        text_itr = iterate_triplet(text_path)
        for triplet in text_itr:
            triplet_list.append(triplet)

    return triplet_list


def dict_to_tuple(encoded):
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    return input_ids, attention_mask


class PairwiseAccuracy(tf.keras.metrics.Mean):
    def __init__(self, name='pairwise_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, s1, s2):
        is_correct = tf.cast(tf.less(s2, s1), tf.float32)
        is_correct_f = tf.reduce_mean(is_correct)
        super(PairwiseAccuracy, self).update_state(is_correct_f)

#
# def build_pairwise_eval_dataset(
#         triplet_list, checkpoint_model_name, batch_size, max_seq_length):
#     c_log.info("build_pairwise_eval_dataset")
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint_model_name)
#     encoded_data = defaultdict(list)
#     for q, d1, d2 in triplet_list:
#         mapping = {
#             "q": q,
#             "d1": d1,
#             "d2": d2
#         }
#         for role, text in mapping.items():
#             encoded = tokenizer(text,
#                     padding="max_length", max_length=max_seq_length)
#             encoded_data[role].append(dict_to_tuple(encoded))
#
#     def get_generator(role) -> Iterable[Tuple]:
#         yield from encoded_data[role]
#
#     eval_dataset = {}
#     for role in pairwise_roles:
#         int_list = tf.TensorSpec([None], dtype=tf.int32)
#         output_signature = (int_list, int_list)
#         dataset = tf.data.Dataset.from_generator(lambda : get_generator(role), output_signature=output_signature)
#         dataset = dataset.batch(batch_size)
#         eval_dataset[role] = dataset
#     return eval_dataset


# each instance is (query, d_pos, d_neg), where each of documents are (input_ids, attention_masks)
def build_pairwise_eval_dataset(
        triplet_list, checkpoint_model_name, batch_size, max_seq_length) -> tf.data.Dataset:
    c_log.info("build_pairwise_eval_dataset")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_model_name)

    def encode(text):
        d = tokenizer(text, padding="max_length", max_length=max_seq_length)
        return dict_to_tuple(d)

    items = []
    for q, d1, d2 in triplet_list:
        e = encode(q), encode(d1), encode(d2)
        items.append(e)

    def get_generator() -> Iterable[Tuple]:
        yield from items

    int_list = tf.TensorSpec([None], dtype=tf.int32)
    int_pair_list = (int_list, int_list)
    output_signature = int_pair_list, int_pair_list, int_pair_list
    dataset = tf.data.Dataset.from_generator(get_generator, output_signature=output_signature)
    dataset = dataset.batch(batch_size)
    return dataset


class PairwiseEval:
    def __init__(self,
                 triplet_encoded: tf.data.Dataset,
                 strategy: Strategy,
                 model: tf.keras.models.Model
                 ):
        self.triplet_encoded = triplet_encoded
        self.strategy = strategy
        self.model = model
        self.metrics = {
            'pairwise_accuracy': PairwiseAccuracy()
        }

    @tf.function
    def eval_fn(self, item):
        q, d1, d2 = item
        q_enc = self.model(q, training=False)
        d1_enc = self.model(d1, training=False)
        d2_enc = self.model(d2, training=False)

        def score(q_enc, d_enc):
            return tf.reduce_sum(tf.multiply(q_enc, d_enc), axis=1)

        s1 = score(q_enc, d1_enc)
        s2 = score(q_enc, d2_enc)

        for m in self.metrics.values():
            m.update_state(s1, s2)

    def do_eval(self):
        c_log.info("PairwiseEval::do_eval")

        num_steps = 2
        iterator = iter(self.triplet_encoded)

        for idx in range(num_steps):
            args = next(iterator),
            per_replica = self.strategy.run(self.eval_fn, args=args)

        metrics = self.metrics
        metric_res = fetch_metric_result(metrics)
        return 0.0, metric_res
