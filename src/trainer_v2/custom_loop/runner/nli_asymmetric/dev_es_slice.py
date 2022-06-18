import sys

import tensorflow as tf
from tensorflow import keras

from cpath import get_bert_config_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import list_equal
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.asymmetric2 import ESSliceSegs
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayer
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    model = ESSliceSegs(FuzzyLogicLayer)
    run_config: RunConfig2 = get_run_config2_nli(args)
    model_config = ModelConfig2SegProject()
    bert_config = load_bert_config(get_bert_config_path())

    model.build_model(bert_config, model_config)

    inputs = model.model.inputs
    output = model.debug_var
    new_model = keras.Model(inputs=inputs, outputs=output, name="bert_model")

    def dataset_factory_a(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    def dataset_factory_b(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, ModelConfig2SegProject(), is_for_training)

    dataset_a = dataset_factory_a(run_config.dataset_config.train_files_path, False)
    dataset_b = dataset_factory_b(run_config.dataset_config.eval_files_path, False)
    iterator_a = iter(dataset_a)
    batch_a = next(iterator_a)
    x_a, y_a = batch_a
    iterator_b = iter(dataset_b)
    batch_b = next(iterator_b)
    x_b, y_b = batch_b

    output_v = new_model(x_a)

    _, _, input_ids2, segment_ids2 = x_a
    input_ids2_1, input_ids2_2 = output_v
    _, _, input_ids2_b, _ = x_b

    n_shift = tf.reduce_sum(tf.cast(tf.not_equal(input_ids2_1, 0), tf.int32), axis=1, keepdims=True)
    input_ids2_2_n_shift = tf.concat([input_ids2_2, -n_shift], axis=1)
    # input_ids2_2_r = tf.map_fn(fn=lambda x: tf.roll(x[:-1], shift=x[-1], axis=0),
    #                            elems=input_ids2_2_n_shift)
    # input_ids2_2_r = tf.roll(input_ids2_2, shift=-n_shift, axis=0)
    def drop_tail(tensor):
        l = tensor.numpy().tolist()
        output = []
        for i, v in enumerate(l):
            if v == 0:
                break
            output.append(v)
        return output

    tokenizer = get_tokenizer()

    for data_idx in range(16):
        print('----')
        orig_a = drop_tail(input_ids2[data_idx])
        after_a_1 = drop_tail(input_ids2_1[data_idx])
        after_a_2 = drop_tail(input_ids2_2[data_idx])
        if not list_equal(orig_a, after_a_1 + after_a_2):
            print("list_equal(orig_a, after_a_1 + after_a_2) wrong")

        print("input_ids2_from_a", orig_a)

        seg2_1 = input_ids2_b[data_idx][:50]
        seg2_2 = input_ids2_b[data_idx][50:]
        after_b_1 = drop_tail(seg2_1)
        after_b_2 = drop_tail(seg2_2)
        print(tokenizer.convert_ids_to_tokens(orig_a))
        after_b = after_b_1[1:-1] + after_b_2[1:-1]
        print("input_ids_from_b", after_b)
        if not list_equal(orig_a[:-1], after_b):
            print("list_equal(orig_a, after_b_c) wrong")
        # print("input_ids2_2_r", input_ids2_2_r[data_idx])
        print(tokenizer.convert_ids_to_tokens(after_b))


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
