import tensorflow as tf
from transformers import AutoTokenizer


from cpath import get_canonical_model_path
from trainer_v2.chair_logging import c_log
import numpy as np
from typing import List, Iterable, Callable, Dict, Tuple, Set

def splade_max(out_vector, mask):
    mask_ex = tf.expand_dims(mask, 2)
    t = tf.math.log(1. + tf.nn.relu(out_vector))
    t = t * mask_ex
    t = tf.reduce_max(t * mask_ex, axis=1)
    return t


class SPLADEWrap:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = tf.keras.models.load_model(model_dir)
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}

    def encode(self, text):
        encoded_input = self.tokenizer(text)
        input_ids = tf.expand_dims(encoded_input["input_ids"], axis=0)
        attention_mask = tf.expand_dims(encoded_input["attention_mask"], axis=0)
        output = self.model({'input_ids': input_ids, 'attention_mask': attention_mask})
        mask = tf.cast((input_ids != 0), tf.float32)
        tf_out = splade_max(output['logits'], mask).numpy()
        return tf_out[0]

    def encode_batch(self, text_list):
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, max_length=512)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        input = {'input_ids': input_ids, 'attention_mask': attention_mask}
        output = self.model(input)
        mask = tf.cast(attention_mask, tf.float32)
        tf_out = splade_max(output['logits'], mask).numpy()
        return tf_out

    def encode_batch_simple_out(self, text_list, threshold=0.1):
        output = []
        for t in self.encode_batch(text_list):
            sparse = get_sparse_representation(t)
            out_t = [(self.reverse_voc[token_id], score) for token_id, score in sparse if score > threshold]
            output.append(out_t)
        return output

    def decode_spare_rep(self, sr:  List[Tuple[int, float]]):
        out = []
        for token_id, score in sr:
            token = self.reverse_voc[token_id]
            out.append((token, score))
        return out


def get_sparse_representation(np_1d) -> List[Tuple[int, float]]:
    indices = np.flatnonzero(np_1d)
    return [(i, np_1d[i]) for i in indices]


def get_splade():
    model_dir = get_canonical_model_path("distilsplade_max_tf")
    return SPLADEWrap(model_dir)
