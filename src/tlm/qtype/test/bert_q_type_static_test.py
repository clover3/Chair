import tensorflow as tf

from cpath import at_data_dir
from models.keras_model.bert_keras.bert_common_eager import get_shape_list_no_name
from models.keras_model.bert_keras.modular_unnamed import BertLayer
from tlm.model_cnfig import JsonConfig
from tlm.qtype.BertQType import BertQTypeQOnly, duplicate_w_q_only_version
from tlm.qtype.qtype_model_fn import set_dropout_to_zero


def main():
    is_training = False
    config_path = at_data_dir("config", "qtype_weights_mlp.json")
    model_config = JsonConfig.from_json_file(config_path)
    model = BertQTypeQOnly()
    model_config_predict = set_dropout_to_zero(model_config)

    input_ids = [101, 3003, 3003, 101] + [2002] * 10 + [101]
    input_ids2 = [101, 3003, 3003, 101] + [2005] * 10 + [101]
    segment_ids = [0] * 4 + [1] * (len(input_ids) - 4)
    input_mask = [1] * len(input_ids)

    input_ids = tf.stack([input_ids, input_ids2], axis=0)
    segment_ids = tf.stack([segment_ids, segment_ids], axis=0)
    input_mask = tf.stack([input_mask, input_mask], axis=0)
    routine(model_config_predict, False, input_ids, input_mask, segment_ids)


def routine(config, is_training, orig_input_ids, orig_input_mask, orig_token_type_ids):
    input_shape = get_shape_list_no_name(orig_input_ids)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    idx_q_only = 0
    idx_full = 1
    print('orig_input_ids', orig_input_ids.shape)
    concat_input_ids, concat_input_mask, concat_token_type_ids = \
        duplicate_w_q_only_version(orig_input_ids, orig_input_mask, orig_token_type_ids)
    print('concat_input_ids', concat_input_ids.shape)
    for j in range(len(concat_input_ids)):
        print(concat_input_ids[j])
    print()
    bert_layer = BertLayer(config, True, True)
    inputs = concat_input_ids, concat_input_mask, concat_token_type_ids
    all_layers = bert_layer(inputs)
    layer8 = all_layers[8]
    print('layer8', layer8.shape)
    emb_layer = bert_layer.embedding_layer.embedding_output
    layers = all_layers
    print('emb_layer', emb_layer.shape)

    def tensor_equal(t1, t2):
        return tf.reduce_sum(tf.square(t1 - t2)).numpy() < 0.0001
    print('layer8', tensor_equal(layer8[0], layer8[1]))
    print('emb_layer', tensor_equal(emb_layer[0], emb_layer[1]))
    all_layers = tf.stack([emb_layer] + layers, 2)  # [batch_size, seq_length, num_layer+1, hidden_dim]
    print(tensor_equal(emb_layer[0], emb_layer[1]))
    print('all_layers', tensor_equal(all_layers[0], all_layers[1]))
    all_layers_paired = tf.reshape(all_layers, [2, batch_size, seq_length, -1, config.hidden_size])
    print('all_layers_paired', tensor_equal(all_layers_paired[0, 0], all_layers_paired[0, 1]))
    q_only = all_layers_paired[idx_q_only]
    is_seg1 = tf.equal(orig_token_type_ids, 0)
    is_seg1_mask = tf.cast(tf.reshape(is_seg1, [batch_size, -1, 1, 1]), tf.float32)
    # all_layers_seg1 = all_layers_seg1 * is_seg1_mask
    maybe_q_len_enough = 64
    q_only = q_only[:, :maybe_q_len_enough, :, :]
    num_layer_plus_one = config.num_hidden_layers + 1
    dim_per_token = num_layer_plus_one * config.hidden_size
    flatten_all_layers = tf.reshape(q_only, [batch_size, -1, dim_per_token])
    print('q_only', tensor_equal(q_only[0], q_only[1]))
    print(q_only.shape)
    print(q_only[0] - q_only[1])


if __name__ == "__main__":
    main()
