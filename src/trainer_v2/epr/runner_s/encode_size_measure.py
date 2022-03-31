from typing import Dict

import tensorflow as tf
from tensorflow import Tensor
from transformers import MPNetTokenizer

from cache import save_list_to_jsonl, save_to_pickle
from trainer_v2.epr.mpnet import TFSBERT
from trainer_v2.epr.s_bert_enc import get_encode_fn, load_segmented_data


def main():
    tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
    model_path = "C:\\work\\Code\\Chair\\output\\model\\runs\\paraphrase-mpnet-base-v2-keras\\model-1"
    config_path = "C:\\work\\Code\\Chair\\output\\model\\runs\\paraphrase-mpnet-base-v2\\config.json"
    sbert = TFSBERT(model_path, config_path)
    encode_fn = get_encode_fn(tokenizer)
    json_iter = load_segmented_data("snli", "validation", 0)
    n_item = 10
    feature_keys = ["input_ids", "attention_mask"]

    save_data = []
    for idx, json_d in enumerate(json_iter):
        if idx >= n_item:
            break
        d: Dict[str, Tensor] = encode_fn(json_d)

        def get_avg_vector(segment):
            item = d[segment]
            t_d = {}
            for feature_name in feature_keys:
                t = tf.ragged.constant(item[feature_name])
                t_d[feature_name] = t.to_tensor()

            tensor = sbert.predict_from_ids(t_d)
            print(tensor.shape)
            return tensor.numpy().tolist()

        v = get_avg_vector('premise')

        save_data.append(v)
        save_data.append(v)

    save_to_pickle(save_data, "encode_size_measure")
    save_list_to_jsonl(save_data, "C:\\work\\Code\\Chair\\output\\temp1")


if __name__ == "__main__":
    main()