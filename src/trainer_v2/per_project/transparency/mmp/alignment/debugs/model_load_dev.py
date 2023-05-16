import sys

from transformers import AutoTokenizer, TFBertMainLayer
from typing import List, Iterable, Callable, Dict, Tuple, Set

from tf_util.lib.tf_funcs import find_layer
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.probe.probe_network import ProbeOnBERT
from trainer_v2.per_project.transparency.mmp.rerank import get_scorer
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_qd_encoder, get_dummy_input_for_bert_layer
import tensorflow as tf


def main():
    model_path = sys.argv[1]
    c_log.info("Building scorer")
    score_fn = get_scorer(model_path)
    q = "who is the president of US"
    d = "The president of US is donald trump"
    print(score_fn([(q, d)]))

def main2():
    model_path = sys.argv[1]
    c_log.info("Building scorer")
    ranking_model = tf.keras.models.load_model(model_path, compile=False)
    network = ProbeOnBERT(ranking_model)
    max_seq_length = 256

    def build_inference_model2(paired_model):
        c_log.info("build_inference_model2")
        input_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids1")
        segment_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids1")
        inputs = [input_ids1, segment_ids1]
        input_1 = {
            'input_ids': input_ids1,
            'token_type_ids': segment_ids1
        }

        old_bert_layer = find_layer(paired_model, "bert")
        dense_layer = find_layer(paired_model, "classifier")
        new_bert_layer = TFBertMainLayer(old_bert_layer._config, name="bert")
        param_values = tf.keras.backend.batch_get_value(old_bert_layer.weights)
        _ = new_bert_layer(get_dummy_input_for_bert_layer())
        tf.keras.backend.batch_set_value(zip(new_bert_layer.weights, param_values))
        bert_output = new_bert_layer(input_1)
        logits = dense_layer(bert_output['pooler_output'])[:, 0]
        new_model = tf.keras.models.Model(inputs=inputs, outputs=logits)
        return new_model

    q = "who is the president of US"
    d = "The president of US is donald trump"
    inference_model = build_inference_model2(network.model)

    qd_encoder = get_qd_encoder(max_seq_length)
    def score_fn(qd_list: List):
        dataset = qd_encoder(qd_list)
        dataset = dataset.batch(16)
        output = inference_model.predict(dataset)
        print(output.shape)
        return output

    print(score_fn([(q, d)]))

def get_dev_batch():
    q = "who is the president of US"
    d = "The president of US is donald trump"
    max_seq_length = 256
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def generator(qd_list):
        for query, document in qd_list:
            encoded_input = tokenizer.encode_plus(
                query,
                document,
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
                return_tensors="tf"
            )
            return {
                "input_ids1": encoded_input['input_ids'],
                "token_type_ids1": encoded_input['token_type_ids'],
                "input_ids2": encoded_input['input_ids'],
                "token_type_ids2": encoded_input['token_type_ids'],
            }

    batch = generator([(q, d)])
    return batch


def main3():
    model_path = sys.argv[1]
    c_log.info("Building scorer")
    model = tf.keras.models.load_model(model_path, compile=False)
    c_log.info("Built scorer")

    batch = get_dev_batch()

    output = model(batch)
    print(output['logits'])



if __name__ == "__main__":
    main3()

