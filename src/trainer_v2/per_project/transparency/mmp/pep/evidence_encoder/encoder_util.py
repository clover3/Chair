from collections import OrderedDict

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.data_gen.base import concat_tuple_windows
from tlm.data_gen.bert_data_gen import create_int_feature, create_float_feature


class MultipleQDSegmentsEncoder:
    def __init__(self, segment_len, n_max_qt, n_max_dt):
        self.segment_len = segment_len
        self.n_max_qt = n_max_qt
        self.n_max_dt = n_max_dt
        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer

    def pack_to_order_dict(self, qt_seq_list, dt_seq_list, scores):
        q_all_input_seg_ids = self.form_windows_concat(qt_seq_list, self.n_max_qt, 0)
        d_all_input_seg_ids = self.form_windows_concat(dt_seq_list, self.n_max_dt, 1)
        scores_flat = np.reshape(scores, [-1])
        n_pad = self.n_max_qt * self.n_max_dt - len(scores_flat)
        scores_list: list[float] = scores_flat.tolist()
        scores_padded = scores_list + [0.] * n_pad

        features = OrderedDict()
        for prefix, input_seg_ids in [("q", q_all_input_seg_ids), ("d", d_all_input_seg_ids)]:
            input_ids, segment_ids = input_seg_ids
            features[f"{prefix}_input_ids"] = create_int_feature(input_ids)
            features[f"{prefix}_segment_ids"] = create_int_feature(segment_ids)
        features['scores'] = create_float_feature(scores_padded)
        return features

    def form_windows_concat(self, token_seq_list, n_max_window, type_id):
        tuple_window_list = []
        for i in range(n_max_window):
            try:
                tokens = token_seq_list[i]
            except IndexError:
                tokens = []

            input_seg_ids = self.format_tokens(tokens, type_id)
            tuple_window_list.append(input_seg_ids)
        all_input_seg_ids = concat_tuple_windows(tuple_window_list, self.segment_len)
        return all_input_seg_ids

    def format_tokens(self, tokens, type_id):
        tokens = ["[CLS]"] + tokens
        segment_ids = [type_id] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids[:self.segment_len]
        segment_ids = segment_ids[:self.segment_len]
        pad_len = self.segment_len - len(input_ids)
        input_ids = input_ids + [0] * pad_len
        segment_ids = segment_ids + [0] * pad_len
        return input_ids, segment_ids