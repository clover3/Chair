import json
import sys
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Any, Iterable
import tensorflow as tf
from typing import List, Iterable, Callable, Dict, Tuple, Set

from tensorflow.python.distribute.values import PerReplica
from transformers import TFBertMainLayer
from tensorflow import keras


from cache import save_to_pickle
from cpath import output_path
from misc_lib import path_join, TimeEstimator
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_qd_encoder, get_dummy_input_for_bert_layer


def load_grad_extraction_model(model_path):
    c_log.info("Loading model from %s", model_path)
    paired_model = tf.keras.models.load_model(model_path, compile=False)
    old_bert_layer = paired_model.layers[4]
    c_log.info("Build new bert_layer")
    bert_layer = TFBertMainLayer(old_bert_layer._config, name="bert")
    param_values = keras.backend.batch_get_value(old_bert_layer.weights)
    # print("bert_layer.weights", bert_layer.weights)
    # print("old_bert_layer.weights", old_bert_layer.weights)

    dummy = get_dummy_input_for_bert_layer()
    _ = bert_layer(dummy)

    # for w1, w2 in zip(old_bert_layer.weights, bert_layer.weights):
    #     print("{} - {}".format(w1.name, w2.name))

    pairs = list(zip(bert_layer.weights, param_values))
    c_log.info("Overwriting weights for {} items".format(len(pairs)))
    keras.backend.batch_set_value(pairs)
    dense_layer = paired_model.layers[6]
    return bert_layer, dense_layer


import numpy as np

@dataclass
class ModelEncoded:
    input_ids: np.array
    token_type_ids: np.array
    logits: np.array
    hidden_states: np.array
    attentions: np.array
    attention_grads: np.array


def reduce(tensor):
    if isinstance(tensor, PerReplica):
        return tf.concat(tensor.values, axis=0)
    else:
        return tensor


class GradExtractor:
    def __init__(self, model_path,
                 batch_size, strategy):
        max_seq_length = 256
        self.encode_qd = get_qd_encoder(max_seq_length)
        bert_layer, dense_layer = load_grad_extraction_model(model_path)
        self.bert_layer = bert_layer
        self.dense_layer = dense_layer
        self.batch_size = batch_size
        self.strategy = strategy

    def encode(self, input_text_pairs) -> Iterable[ModelEncoded]:
        dataset = self.encode_qd(input_text_pairs)
        dataset = dataset.batch(self.batch_size)
        dist_dataset = distribute_dataset(self.strategy, dataset)
        for batch in dist_dataset:
            (input_ids, token_type_ids), = batch
            output = self.strategy.run(self.encode_fn, args=(input_ids, token_type_ids,))
            attention_grads, attentions, hidden_states, logits = output
            batch_size = len(reduce(input_ids))
            vars = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'logits': logits,
                'hidden_states': hidden_states,
                'attentions': attentions,
                'attention_grads': attention_grads,
            }
            vars = {k: reduce(v) for k, v in vars.items()}
            for i in range(batch_size):
                var_per_i = {k: v[i].numpy() for k, v in vars.items()}
                me = ModelEncoded(**var_per_i)
                yield me

    @tf.function
    def encode_fn(self, input_ids, token_type_ids):
        batch_d = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids
        }
        # with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
        with tf.GradientTape() as tape:
            outputs = self.bert_layer(
                batch_d,
                output_attentions=True,
                output_hidden_states=True)
            attn = outputs.attentions
            hidden = outputs.hidden_states
            pooled_output = outputs['pooler_output']
            logits = self.dense_layer(pooled_output)
            tape.watch(attn)
        attention_grads = tape.gradient(logits, attn)
        hidden_states = tf.stack(hidden, axis=1)
        attentions = tf.stack(attn, axis=1)
        attention_grads = tf.stack(attention_grads, axis=1)
        return attention_grads, attentions, hidden_states, logits


def extract_save_align(
        compute_alignment_fn: Callable[[ModelEncoded], Any],
        qd_itr, strategy,
        save_path, num_record,
        model_save_path, batch_size,
    ):
    ticker = TimeEstimator(num_record)
    out_f = open(save_path, "w")
    c_log.info("{}".format(strategy))
    tf.debugging.set_log_device_placement(True)
    with strategy.scope():
        extractor = GradExtractor(
            model_save_path,
            batch_size,
            strategy
        )
        me_itr: Iterable[ModelEncoded] = extractor.encode(qd_itr)
        for me in me_itr:
            aligns = compute_alignment_fn(me)
            logits = me.logits.tolist()
            out_info = {'logits': logits, 'aligns': aligns}
            out_f.write(json.dumps(out_info) + "\n")
            ticker.tick()


def main():
    # input_text_pairs = [("query", "document")]
    query = 'When Does Fear The Walking Dead Premiere'
    doc_pos = "Finally, some real footage from Fear The Walking Dead ! Yes, we've seen the little teasers and gotten glimpses at the characters, but now we have an honest to goodness trailer and premiere date! Fear The Walking Dead premieres Sunday, Aug. 23 at 9 p.m. on AMC with a 90-minute premeire seen around the world."
    # doc_pos = "Fear The Walking Dead Premiere is Sunday"
    input_text_pairs = [(query, doc_pos)]
    extractor = GradExtractor(sys.argv[1])
    itr = extractor.encode(input_text_pairs)
    item = list(itr)[0]
    save_to_pickle(item, "attn_grad_dev")


if __name__ == "__main__":
    main()

