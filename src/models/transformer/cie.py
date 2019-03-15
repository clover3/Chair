from task.transformer_est import Transformer, Classification
from models.transformer import bert
import tensorflow as tf
from data_generator.NLI import nli
from trainer import tf_module


class token_regression:
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
        scores = tf.placeholder(tf.int32, [None, seq_length])

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

        enc = self.model.get_sequence_output()
        enc = tf.layers.dense(enc, hp.hidden_units, name="dense1") # [ None, seq_length, hidden]
        logits = tf.layers.dense(enc, 1, name="dense2")
        self.logits = tf.reshape(logits, [-1, seq_length])

        self.sout = tf.sigmoid(self.logits)
        #self.sout = tf.nn.softmax(self.logits, axis=1)
        #losses = tf.cast(self.y, tf.float32) * -tf.log(self.sout) # [ None, seq_length ]

        self.loss = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(self.y, logits=self.logits))
        tf.summary.scalar('loss', self.loss)

        p = self.sout
        pred = tf.less(tf.zeros_like(p), p - 0.5)
        self.prec = tf_module.precision_b(pred, self.y)
        self.recall = tf_module.recall_b(pred, self.y)
        tf.summary.scalar('prec', self.prec)



class span_selection:
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
        self.begin = tf.placeholder(tf.int32, [None, seq_length])
        self.end = tf.placeholder(tf.int32, [None, seq_length])

        self.y = tf.stack([self.begin, self.end], axis=2)

        self.x_list = [input_ids, input_mask, segment_ids]

        use_one_hot_embeddings = use_tpu
        is_training = True
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        enc = self.model.get_sequence_output()
        enc = tf.layers.dense(enc, hp.hidden_units, name="dense1") # [ None, seq_length, hidden]
        self.logits = tf.layers.dense(enc, 2, name="dense2")

        self.sout = tf.nn.softmax(self.logits, axis=1)
        losses = tf.cast(self.y, tf.float32) * -tf.log(self.sout) # [ None, seq_length ]

        self.loss = tf.reduce_sum(losses)
        tf.summary.scalar('loss', self.loss)

