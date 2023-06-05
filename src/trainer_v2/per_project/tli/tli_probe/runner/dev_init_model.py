import sys
from cpath import get_bert_config_path, get_canonical_model_path2
from taskman_client.wrapper3 import report_run3
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfig300_3
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
import tensorflow as tf

from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.per_project.tli.model_load_h5 import load_weights_from_hdf5
from trainer_v2.train_util.arg_flags import flags_parser


def name_mapping(name, prefix):
    return name


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
    load_weights_from_hdf5(model, h5py_file_path, name_mapping, 197 + 4)
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
