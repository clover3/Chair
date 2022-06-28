import tensorflow as tf
from tensorflow.python.keras import backend as K

from cpath import get_bert_config_path
from trainer_v2.bert_for_tf2.w_mask.attention import AttentionLayerWMask
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.modeling_common.network_utils import ChunkAttentionMaskLayerFreeP


# noinspection PyUnusedLocal
def call(attn_layer, inputs, attention_mask, training=None, **kwargs):
    from_tensor = inputs
    to_tensor = inputs
    #  from_tensor shape - [batch_size, from_seq_length, from_width]
    input_shape = tf.shape(input=from_tensor)
    batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]
    to_seq_len = from_seq_len

    # [B, F, N*H] -> [B, N, F, H]
    def transpose_for_scores(input_tensor, seq_len):
        output_shape = [batch_size, seq_len,
                        attn_layer.params.num_heads, attn_layer.params.size_per_head]
        output_tensor = K.reshape(input_tensor, output_shape)
        return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,F,H]

    query = attn_layer.query_layer(from_tensor)  # [B,F, N*H] [batch_size, from_seq_len, N*H]
    key = attn_layer.key_layer(to_tensor)  # [B,T, N*H]
    value = attn_layer.value_layer(to_tensor)  # [B,T, N*H]

    query = transpose_for_scores(query, from_seq_len)  # [B, N, F, H]
    key = transpose_for_scores(key, to_seq_len)  # [B, N, T, H]

    attention_scores = tf.matmul(query, key, transpose_b=True)  # [B, N, F, T]
    attention_scores = attention_scores / tf.sqrt(float(attn_layer.params.size_per_head))

    if attention_mask is not None:
        attention_mask = tf.expand_dims(attention_mask, axis=1)  # [B, 1, F, T]
        # {1, 0} -> {0.0, -inf}
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * attn_layer.params.negative_infinity
        attention_scores = tf.add(attention_scores, adder)  # adding to softmax -> its like removing them entirely

    # scores to probabilities
    attn_layer.attention_scores = attention_scores
    attention_probs = tf.nn.softmax(attention_scores)  # [B, N, F, T]
    attn_layer.attention_probs = attention_probs
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = attn_layer.dropout_layer(attention_probs,
                                               training=training)  # [B, N, F, T]
    print("attention_probs", attention_probs)
    # [B,T,N,H]
    value = tf.reshape(value, [batch_size, to_seq_len,
                               attn_layer.params.num_heads, attn_layer.params.size_per_head])
    # [B, N, F, T] * [B,N,T,H] -> [B, N, F, H]
    value = tf.transpose(a=value, perm=[0, 2, 1, 3])  # [B, N, T, H]

    # [B,N,T,H]
    context_layer = tf.matmul(attention_probs, value)  # [B, N, F, H]
    # [B, N, F, T] * [B,N,T,H] -> [B, F, N, H]
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])  # [B, F, N, H]

    # [B, N, F, T] * [B,N,T,H] -> [B, F, N * H]
    output_shape = [batch_size, from_seq_len,
                    attn_layer.params.num_heads * attn_layer.params.size_per_head]
    context_layer = tf.reshape(context_layer, output_shape)
    return attention_probs


def main():
    params = load_bert_config(get_bert_config_path())
    size_per_head = params.hidden_size // params.num_heads
    attn_layer = AttentionLayerWMask.from_params(params,
                                                 size_per_head=size_per_head)
    B = 2
    L = 20
    H = params.hidden_size
    inputs = tf.random.uniform([B, L, H])
    p_array = tf.ones([2, 10], tf.int32)
    h_array = tf.ones([2, 10], tf.int32)
    attention_mask = ChunkAttentionMaskLayerFreeP()([p_array, h_array])
    attn_layer(inputs, attention_mask)
    attention_probs = call(attn_layer, inputs, attention_mask)

    attention_probs = attention_probs[0]

    head_no = 0

    p_idx = 2
    h_idx = 10 + 2
    print(f"from={h_idx}, to={h_idx}, {attention_probs[head_no, h_idx, h_idx]} > 0")
    print(f"from={p_idx}, to={h_idx}, {attention_probs[head_no, p_idx, h_idx]} == 0")
    print(f"from={h_idx}, to={p_idx}, {attention_probs[head_no, h_idx, p_idx]} > 0")
    print(f"from={p_idx}, to={p_idx}, {attention_probs[head_no, p_idx, p_idx]} > 0")



if __name__ == "__main__":
    main()