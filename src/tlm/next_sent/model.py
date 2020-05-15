import tensorflow as tf

from models.transformer import bert
from tf_v2_support import placeholder


class transformer_next_sent:
    def __init__(self, hp, num_classes, voca_size, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False

        input_ids = placeholder(tf.int64, [None, seq_length])
        input_mask = placeholder(tf.int64, [None, seq_length])
        segment_ids = placeholder(tf.int64, [None, seq_length])
        label_ids = placeholder(tf.int64, [None])
        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pooled_output = self.model.get_pooled_output()
        with tf.variable_scope("cls/seq_relationship"):
            output_weights = tf.get_variable(
                "output_weights", [num_classes, hp.hidden_units],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )

            output_bias = tf.get_variable(
                "output_bias", [num_classes],
                initializer=tf.zeros_initializer()
            )

            logits = tf.matmul(pooled_output, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)
        loss = tf.reduce_mean(input_tensor=loss_arr)

        self.loss = loss
        self.logits = logits
        self.sout = tf.nn.softmax(self.logits)

    def batch2feed_dict(self, batch):
        if len(batch) == 3:
            x0, x1, x2 = batch
            feed_dict = {
                self.x_list[0]: x0,
                self.x_list[1]: x1,
                self.x_list[2]: x2,
            }
        else:
            x0, x1, x2, y = batch
            feed_dict = {
                self.x_list[0]: x0,
                self.x_list[1]: x1,
                self.x_list[2]: x2,
                self.y: y,
            }
        return feed_dict
