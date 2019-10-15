from task.transformer_est import Transformer, Classification
from models.transformer import bert
import tensorflow as tf
from trainer import tf_module

class crs_transformer:
    def __init__(self, hp, voca_size, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        s_portion = tf.placeholder(tf.float32, [None])
        d_portion = tf.placeholder(tf.float32, [None])
        s_sum = tf.placeholder(tf.int64, [None])
        d_sum = tf.placeholder(tf.int64, [None])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = [s_portion, d_portion]
        self.y_sum = [s_sum, d_sum]


        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        enc = self.model.get_sequence_output()
        pool = tf.layers.dense(enc[:,0,:], hp.hidden_units, name="pool")

        s_logits = tf.layers.dense(pool, 2, name="cls_dense_support")
        d_logits = tf.layers.dense(pool, 2, name="cls_dense_dispute")

        loss = 0
        self.acc = []
        for logits, y, mask_sum in [(s_logits, self.y[:, 0], s_sum),
                                    (d_logits, self.y[:, 1], d_sum)]:
            labels = tf.cast(tf.greater(y, 0.5), tf.int32)
            labels = tf.one_hot(labels, 2)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            acc = tf_module.accuracy(logits, self.y[2])

            self.acc.append(acc)
            tf.summary.scalar("acc", self.acc)

            loss_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=labels)

            loss_arr = loss_arr * mask_sum
            loss += tf.reduce_sum(loss_arr)

        self.loss = loss

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('s_acc', self.acc[0])
        tf.summary.scalar('d_acc', self.acc[1])



