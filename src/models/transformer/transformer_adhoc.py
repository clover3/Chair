from task.transformer_est import Transformer, Classification
from models.transformer import bert
import tensorflow as tf
from data_generator.NLI import nli
from trainer import tf_module

class transformer_adhoc:
    def __init__(self, hp, voca_size, mode = 1):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(nli.num_classes)

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        scores = tf.placeholder(tf.float32, [None])
#        self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = scores

        use_one_hot_embeddings = use_tpu
        is_training = True
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode == 1:
            enc = self.model.get_pooled_output()
        else:
            enc = self.model.get_all_encoder_layers()
        logits = tf.layers.dense(enc, 1, name="reg_dense") # [ None, 1]
        self.logits = logits

        paired = tf.reshape(logits, [-1, 2])
        y_paired = tf.reshape(self.y, [-1,2])
        raw_l = (paired[:, 1] - paired[:, 0])
        losses = tf.maximum(hp.alpha - (paired[:, 1] - paired[:, 0]) , 0)

        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', self.loss)
