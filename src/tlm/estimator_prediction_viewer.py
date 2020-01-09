import os
import pickle
from functools import partial

import numpy as np

from cpath import output_path
from data_generator.common import get_tokenizer
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import Cell


def flatten_batches(data):
    keys = list(data[0].keys())
    vectors = {}
    for e in data:
        for key in keys:
            if key not in vectors:
                vectors[key] = []
            vectors[key].append(e[key])
    for key in keys:
        vectors[key] = np.concatenate(vectors[key], axis=0)
    return vectors


class EstimatorPredictionViewer:
    class Entry:
        def __init__(self, idx, viewer):
            self.idx = idx
            self.viewer = viewer

            for method in self.viewer.method_list:
                if method.startswith("get_"):
                    method_fn = getattr(EstimatorPredictionViewer, method)

                    wrap_fn = partial(method_fn, self.viewer, self.idx)
                    setattr(self, method, wrap_fn)


    def __init__(self, filename):
        self.vectors, self.keys, self.data_len = self.estimator_prediction_loader(filename)
        self.tokenizer = get_tokenizer()
        self.method_list = list([func for func in dir(EstimatorPredictionViewer)
                            if callable(getattr(EstimatorPredictionViewer, func))])

    def __iter__(self):
        for i in range(self.data_len):
            yield self.Entry(i, self)

    def get_tokens(self, idx, key):
        return self.tokenizer.convert_ids_to_tokens(self.vectors[key][idx])

    def get_vector(self, idx, key):
        return self.vectors[key][idx]

    @staticmethod
    def estimator_prediction_loader(filename):
        p = os.path.join(output_path, filename)
        data = pickle.load(open(p, "rb"))

        keys = list(data[0].keys())
        vectors = flatten_batches(data)

        any_key = keys[0]
        data_len = len(vectors[any_key])

        return vectors, keys, data_len

    def get_mask_resolved_input_mask_with_input(self, idx, key=""):
        tokens = self.get_tokens(idx, "input_ids")
        masked_tokens = self.get_tokens(idx, "masked_input_ids")
        if key :
            values = self.vectors[key][idx]
        for i in range(len(tokens)):
            if masked_tokens[i] == "[MASK]":
                if key :
                    value_str = float_aware_strize(values[i])
                    masked_tokens[i] = "[{}:{}]".format(tokens[i], value_str)
                else:
                    masked_tokens[i] = "[{}]".format(tokens[i])
            if tokens[i] == "[SEP]":
                tokens[i] = "[SEP]<br>"

        return masked_tokens

    def cells_from_tokens(self, tokens, scores=None, stop_at_pad=True):
        return cells_from_tokens(tokens, scores, stop_at_pad)

    def cells_from_anything(self, vectors, scores=None):
        cells = []
        for i, v in enumerate(vectors):
            h_score = scores[i] if scores is not None else 0
            cells.append(Cell(float_aware_strize(v), h_score))
        return cells

    def cells_from_scores(self, scores, hightlight=True):
        cells = []
        for i, score in enumerate(scores):
            h_score = scores if hightlight else 0
            cells.append(Cell(float_aware_strize(score), h_score))
        return cells


def float_aware_strize(obj):
    try:
        i = int(obj)
        v = float(obj)

        if abs(i-v) < 0.0001: # This is int
            return str(obj)
        else:
            return "{:04.2f}".format(obj)
    except:
        return str(obj)

    return str(v)