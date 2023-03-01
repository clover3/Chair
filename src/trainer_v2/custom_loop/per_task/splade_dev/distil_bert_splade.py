import os
import pickle
import sys
import re
import h5py
from bert.loader import bert_prefix
import numpy as np

from cache import save_to_pickle, load_from_pickle
from cpath import get_bert_config_path, data_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, ModelConfig300_3, BertClassifier
from trainer_v2.custom_loop.per_task.trainer import Trainer
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf
import keras.backend


def load_weights_from_hdf5(model, h5_path, map_to_stock_fn, n_expected_restore):
    # stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())
    param_storage = h5py.File(h5_path, 'r')
    prefix = "bert"
    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []
    bert_params = model.weights
    param_values = keras.backend.batch_get_value(model.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_to_stock_fn(param.name, prefix)

        if stock_name in param_storage:
            ckpt_value = param_storage[stock_name]
            if "kernel" in param.name:
                ckpt_value = np.transpose(ckpt_value)
            if param_value.shape != ckpt_value.shape:
                c_log.warn("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                           "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                      stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            c_log.info("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, h5_path))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)
    if n_expected_restore is not None and n_expected_restore == len(weight_value_tuples):
        pass
    else:
        msg = "Done loading {} BERT weights from checkpoint into {} (prefix:{}). " \
                   "Count of weights not found in the checkpoint was: [{}]. " \
                   "Count of weights with mismatched shape: [{}]".format(
            len(weight_value_tuples), model, prefix, skip_count, len(skipped_weight_value_tuples))
        c_log.warning(msg)

        param_storage_keys = set(param_storage.keys())
        c_log.warning("Unused weights from checkpoint: %s",
                   "\n\t" + "\n\t".join(sorted(param_storage_keys.difference(loaded_weights))))
        raise ValueError("Checkpoint load exception")

    return skipped_weight_value_tuples


def name_mapping(source_key, prefix="bert"):
    conversion_reg_list = [
        (prefix, "distilbert"),
        (r"encoder", "transformer"),
        (r"layer_(\d)", r"layer.\1"),
        (r"attention/output/LayerNorm", "sa_layer_norm"),
        (r"attention/self/key", "attention.k_lin"),
        (r"attention/self/query", "attention.q_lin"),
        (r"attention/self/value", "attention.v_lin"),
        (r"attention/output/dense", "attention.out_lin"),
        (r"intermediate", "ffn.lin1"),
        (r"(\d)/output/dense", r"\1/ffn.lin2"),
        (r"(\d)/output/LayerNorm", r"\1/output_layer_norm"),
        ("_embeddings/embeddings:0", "_embeddings.weight"),
        ("gamma:0", "weight"),
        ("beta:0", "bias"),
        ("kernel:0", "weight"),
        ("bias:0", "bias")
    ]
    # bert -> distilbert
    # encoder -> transformer
    # layer_5 -> layer.5
    # attention/output/LayerNorm -> sa_layer_norm
    # attention/self/key/ -> attention.k_lin
    # attention/self/query/ -> attention.q_lin
    # attention/self/value/ -> attention.v_lin
    # attention/output/dense -> attention.out_lin

    # intermediate -> ffn.lin1
    # output/dense -> ffn.lin2
    # output/LayerNorm -> output_layer_norm


    # vocab_layer_norm
    # word_embeddings/embeddings:0 -> word_embeddings.weight
    # position_embeddings/embeddings:0 -> position_embeddings.weight
    # token_type_embeddings/embeddings:0  -> ?????
    # LayerNorm/gamma:0 -> LayerNorm.weight
    # LayerNorm/beta:0 -> LayerNorm.bias
    # kernel:0 -> weight
    # bias:0 -> bias

    s = source_key
    for pattern, replacee in conversion_reg_list:
        s = re.sub(pattern, replacee, s)

    s = s.replace("/", ".")
    return s


def splade_max(out_vector, mask):
    print("type(mask)", mask.dtype)
    mask_ex = tf.expand_dims(mask, 2)
    t = tf.math.log(1. + tf.nn.relu(out_vector))
    print("type(t)", t.dtype)
    print("type(mask_ex)", mask_ex.dtype)
    t = t * mask_ex
    print()
    t = tf.reduce_max(t * mask_ex, axis=1)
    return t


def vocab_vector_to_text(vector):
    vocal_len = len(vector)
    scores = []
    for i in range(vocal_len):
        v = vector[i]
        if abs(v) > 0.01:
            scores.append("{0}: {1:.2f}".format(i, v))
    return " ".join(scores)

def main(args):
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()

    checkpoint_path = run_config.train_config.init_checkpoint
    config_path = os.path.join(data_path, "config", 'distilbert.json')
    bert_params = load_bert_config(config_path)
    max_seq_len = bert_params.max_position_embeddings
    l_bert = BertModelLayer.from_params(bert_params, name="bert")
    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    seq_out = l_bert(l_input_ids)  # [batch_size, max_seq_len, hidden_size]
    Dense = tf.keras.layers.Dense
    LayerNorm = tf.keras.layers.LayerNormalization
    vocab_transform = Dense(bert_params.hidden_size, name="vocab_transform")(seq_out)
    vocab_layer_norm = LayerNorm(name='vocab_layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)(vocab_transform)
    vocab_projector = Dense(bert_params.vocab_size, name="vocab_projector")(vocab_layer_norm)
    model = keras.Model(inputs=(l_input_ids, ), outputs=vocab_projector, name="bert_model")
    skipped_weight_value_tuples = load_weights_from_hdf5(
        model, checkpoint_path, name_mapping, n_expected_restore=106)
    print(skipped_weight_value_tuples)
    query = "what is thermal stress?"
    tokenizer = get_tokenizer()
    q_tokens = tokenizer.tokenize(query)
    tokens = ["[CLS]"] + q_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(input_ids)
    input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
    input_ids = np.expand_dims(np.array(input_ids), 0)
    output = model(input_ids)
    mask = tf.cast((input_ids != 0), tf.float32)
    print("mask", mask.dtype)
    tf_out = splade_max(output, mask).numpy()

    torch_out = load_from_pickle("torch_splade_out")
    print("tf_out", vocab_vector_to_text(tf_out[0]))
    print("torch_out", vocab_vector_to_text(torch_out))
    error = np.sum(tf_out - torch_out)
    print("error: {}".format(error))


def main(args):
    from transformers import TFAutoModelForMaskedLM, AutoTokenizer
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()

    config_path = os.path.join(data_path, "config", 'distilbert.json')
    model_type_or_dir = "C:\work\code\chair\output\model\\runs\distilsplade_max"
    model, loading_info = TFAutoModelForMaskedLM.from_pretrained(
        model_type_or_dir, from_pt=True, output_loading_info=True)
    # TFDistilBertForMaskedLM
    print("Model", model)
    model.summary()
    query = "what is thermal stress?"
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    q_tokens = tokenizer(query)
    print("q_tokens", q_tokens)
    # tokens = ["[CLS]"] + q_tokens + ["[SEP]"]
    # input_ids = input_ids + [0] * (10 - len(input_ids))
    # input_ids = np.expand_dims(np.array(input_ids), 0)
    print("input_ids", q_tokens["input_ids"])
    print("attention_mask", q_tokens["attention_mask"])
    print(model.inputs)
    input_ids = tf.expand_dims(q_tokens["input_ids"], axis=0)
    output = model.call(input_ids=input_ids, )
    mask = tf.cast(tf.expand_dims(q_tokens['attention_mask'], axis=0), tf.float32)
    tf_out = splade_max(output.logits, mask).numpy()
    torch_out = load_from_pickle("torch_splade_out")

    print("tf____out", vocab_vector_to_text(tf_out[0]))
    print("torch_out", vocab_vector_to_text(torch_out))
    error = np.sum(tf_out - torch_out)
    print("error: {}".format(error))


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


