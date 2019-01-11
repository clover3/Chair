from task.transformer_est import Transformer, Classification
from models.transformer import bert
import tensorflow as tf
from data_generator.NLI import nli
from trainer import tf_module

class transformer_controversy:
    def __init__(self, hp, voca_size):
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
        score = tf.placeholder(tf.float32, [None])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = score

        use_one_hot_embeddings = use_tpu
        is_training = True
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        enc = self.model.get_pooled_output()
        self.logits = tf.layers.dense(enc, 1, name="reg_dense") # [ None, 1]

        paired = tf.reshape(self.logits, [-1, 2])
        y_paired = tf.reshape(self.y, [-1,2])
        raw_l = (paired[:, 1] - paired[:, 0])
        losses = tf.maximum(hp.alpha - (paired[:, 1] - paired[:, 0]) , 0)
        self.loss = tf.reduce_mean(losses)

        gain = tf.maximum(paired[:, 1] - paired[:, 0], 0)
        self.acc = tf.cast(tf.count_nonzero(gain), tf.float32) / tf.reduce_sum(tf.ones_like(gain))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc', self.acc)



class transformer_controversy_fixed_encoding:
    def __init__(self, hp, voca_size):
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
        score = tf.placeholder(tf.float32, [None])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = score

        use_one_hot_embeddings = use_tpu
        is_training = True
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        all_layers = self.model.get_all_encoder_layers()
        enc = tf.concat(all_layers, axis=2) # [None, seq_len, Hidden_dim * num_blocks]
        per_token_score = tf.layers.dense(enc[0], 1, name="reg_dense") # [ None, seq_len, 1]
        self.logits = tf.reduce_sum(per_token_score, axis=1) # [ None, 1]

        paired = tf.reshape(self.logits, [-1, 2])
        y_paired = tf.reshape(self.y, [-1,2])
        raw_l = (paired[:, 1] - paired[:, 0])
        losses = tf.maximum(hp.alpha - (paired[:, 1] - paired[:, 0]) , 0)
        self.loss = tf.reduce_mean(losses)

        gain = tf.maximum(paired[:, 1] - paired[:, 0], 0)
        self.acc = tf.cast(tf.count_nonzero(gain), tf.float32) / tf.reduce_sum(tf.ones_like(gain))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc', self.acc)

