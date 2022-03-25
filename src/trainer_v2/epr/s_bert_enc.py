import os
from typing import List, Dict

import tensorflow as tf
from tensorflow import Tensor
from transformers import MPNetTokenizer

from cache import load_list_from_jsonl, save_list_to_jsonl
from misc_lib import TimeEstimator
from trainer_v2.epr.mpnet import TFSBERT
from trainer_v2.epr.path_helper import get_segmented_data_path


def align(p_vectors, h_vectors):
    norm_p = tf.nn.l2_normalize(p_vectors, -1)
    norm_h = tf.nn.l2_normalize(h_vectors, -1)
    score_matrix = tf.matmul(norm_p, norm_h, transpose_b=True)
    max_p_idx_for_h = tf.argmax(score_matrix, axis=0)
    max_h_idx_for_p = tf.argmax(score_matrix, axis=1)
    return max_p_idx_for_h, max_h_idx_for_p


def load_segmented_data(dataset_name, split, job_id) -> List[Dict]:
    file_path = get_segmented_data_path(dataset_name, split, job_id)
    return load_list_from_jsonl(file_path, lambda x: x)


def get_encode_fn(tokenizer):
    segment_keys = ["premise", "hypothesis"]
    def encode_fn(e):
        def convert_text_list(text_list):
            # text_list = pad_items(text_list)
            if len(text_list) == 0:
                text_list = [""]
            input_ids = tokenizer(text_list)
            return input_ids
        d = {
            'label': e['label']
        }
        for segment in segment_keys:
            d[segment] = convert_text_list(e[segment])
        return d
    return encode_fn


class EncodeWorker:
    def __init__(self, save_dir, config_path, model_path, jsonl_file_load_fn):
        tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
        self.sbert = TFSBERT(model_path, config_path)
        self.encode_fn = get_encode_fn(tokenizer)
        self.save_dir = save_dir
        self.jsonl_file_load_fn = jsonl_file_load_fn

    def solve(self, json_iter):
        feature_keys = ["input_ids", "attention_mask"]
        out_e_list = []
        ticker = TimeEstimator(len(json_iter))
        for json_d in json_iter:
            d: Dict[str, Tensor] = self.encode_fn(json_d)
            ticker.tick()
            def get_avg_vector(segment):
                item = d[segment]
                t_d = {}
                for feature_name in feature_keys:
                    t = tf.ragged.constant(item[feature_name])
                    t_d[feature_name] = t.to_tensor()
                return self.sbert.predict_from_ids(t_d)

            p_avg_vectors = get_avg_vector('premise')
            h_avg_vectors = get_avg_vector('hypothesis')
            max_p_idx_for_h, max_h_idx_for_p = align(p_avg_vectors, h_avg_vectors)
            json_d['max_p_idx_for_h'] = max_p_idx_for_h.numpy().tolist()
            json_d['max_h_idx_for_p'] = max_h_idx_for_p.numpy().tolist()
            out_e_list.append(json_d)
        return out_e_list

    def work(self, job_id):
        save_path = os.path.join(self.save_dir, str(job_id))
        print("{} will ")
        json_iter = self.jsonl_file_load_fn(job_id)
        out_e_list = self.solve(json_iter)
        save_list_to_jsonl(out_e_list, save_path)