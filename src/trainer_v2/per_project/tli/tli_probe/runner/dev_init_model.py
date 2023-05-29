import sys

from cpath import get_bert_config_path, get_canonical_model_path2
from taskman_client.wrapper3 import report_run3
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfig300_3
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, load_stock_weights
import tensorflow as tf
import h5py
from tensorflow import keras

from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser


def name_mapping(name, prefix):
    return name


def load_weights_from_hdf5(model, h5_path, map_to_stock_fn, n_expected_restore):
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
            if param_value.shape != ckpt_value.shape:
                c_log.warn("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                           "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                      stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            c_log.info("loader: No value for:[{}]".format(param.name))
            skip_count += 1

    keras.backend.batch_set_value(weight_value_tuples)
    if n_expected_restore is not None:
        if n_expected_restore == len(weight_value_tuples):
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


def load_nli_14(config, h5py_file_path):
    num_classes = config.num_classes
    max_seq_len = config.max_seq_length

    bert_params = load_bert_config(get_bert_config_path())
    num_layer = bert_params.num_layers
    bert_params.out_layer_ndxs = list(range(num_layer))
    l_input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    l_token_type_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")
    l_bert = BertModelLayer.from_params(bert_params, name="bert")
    bert_output = l_bert([l_input_ids, l_token_type_ids])
    seq_out = bert_output[-1]
    first_token = seq_out[:, 0, :]
    pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
    pooled = pooler(first_token)
    classifier = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    output = classifier(pooled)
    output = tf.argmax(output, axis=1)
    model = tf.keras.models.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
    load_weights_from_hdf5(model, h5py_file_path, name_mapping, 197+4)
    return model


@report_run3
def main(args):
    c_log.info("dev init model")
    run_config: RunConfig2 = get_run_config2_nli(args)
    input_files = run_config.dataset_config.eval_files_path
    model_config = ModelConfig300_3()

    dataset = get_classification_dataset(input_files, run_config, model_config, False)

    checkpoint_path = get_canonical_model_path2("nli14_0", "model_12500.h5py")
    model = load_nli_14(model_config, checkpoint_path)
    model.compile(metrics=[tf.keras.metrics.Accuracy()])
    batches = dataset.take(10)
    ret = model.evaluate(batches)
    print(ret)



if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
