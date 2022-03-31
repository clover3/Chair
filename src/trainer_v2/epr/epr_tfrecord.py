import collections
from typing import List
from typing import NamedTuple, OrderedDict

import tensorflow as tf

from data_generator.create_feature import create_int_feature


class PairedVectors(NamedTuple):
    p_paired_vectors: List[List]
    h_paired_vectors: List[List]
    p_not_paired_vectors: List[List]
    h_not_paired_vectors: List[List]


def encode(pv: PairedVectors, label: int) -> OrderedDict:
    def encode_vector_list(vector_list):
        tensor = tf.constant(vector_list).to_tensor()
        serialized_nonscalar = tf.io.serialize_tensor(tensor)
        feature = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[serialized_nonscalar.numpy()]))
        return feature

    features = collections.OrderedDict()
    features['p_paired_vectors'] = encode_vector_list(pv.p_paired_vectors)
    features['h_paired_vectors'] = encode_vector_list(pv.h_paired_vectors)
    features['p_not_paired_vectors'] = encode_vector_list(pv.p_not_paired_vectors)
    features['h_not_paired_vectors'] = encode_vector_list(pv.h_not_paired_vectors)
    features['label'] = create_int_feature([label])
    return features
