from collections import OrderedDict

import numpy as np
import tensorflow as tf

from data_generator.create_feature import create_float_feature


def make_tf_feature(x, y, shape) -> OrderedDict:
    x_slice = x[:shape[0], :shape[1], :, :]
    y_slice = y[:shape[0], :shape[1]]

    n_pad1 = shape[0] - x_slice.shape[0]
    n_pad2 = shape[1] - x_slice.shape[1]

    x_padded = np.pad(x_slice, [(0, n_pad1), (0, n_pad2), (0, 0), (0, 0)])
    y_padded = np.pad(y_slice, [(0, n_pad1), (0, n_pad2)])

    def encode_np_array(np_array):
        np_flat = np.reshape(np_array, [-1])
        return create_float_feature(np_flat)

    features = OrderedDict()
    features['x'] = encode_np_array(x_padded)
    features['y'] = encode_np_array(y_padded)
    return features