import tensorflow as tf

import data_generator.NLI.nli_info
from models.transformer import bert
from task.transformer_est import Classification


class transformer_paired:
    def __init__(self, hp, voca_size, method, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(data_generator.NLI.nli_info.num_classes)
        task2_num_classes = 3

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
        if method in [0,1,3,4,5,6]:
            self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        elif method in [2]:
            self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids
        self.y1 = tf.placeholder(tf.int64, [None], name="y1")
        self.y2 = tf.placeholder(tf.int64, [None], name="y2")
        self.f_loc1 = tf.placeholder(tf.int64, [None], name="f_loc1")
        self.f_loc2 = tf.placeholder(tf.int64, [None], name="f_loc2")

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pred, loss = task.predict(self.model.get_sequence_output(), label_ids, True)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc
        #tf.summary.scalar('loss', self.loss)
        #tf.summary.scalar('acc', self.acc)

        enc = self.model.get_sequence_output() # [Batch, Seq_len, hidden_dim]

        logits_raw = tf.layers.dense(enc, 3) # [Batch, seq_len, 3]
        def select(logits, f_loc):
            mask = tf.reshape(tf.one_hot(f_loc, seq_length), [-1,seq_length, 1]) # [Batch, seq_len, 1]
            t = tf.reduce_sum(logits * mask, axis=1)
            return t

        logits1 = select(logits_raw, self.f_loc1) # [Batch, 3]
        logits2 = select(logits_raw, self.f_loc2)  # [Batch, 3]
        self.logits1 = logits1
        self.logits2 = logits2
        label1 = tf.one_hot(self.y1, task2_num_classes) # [Batch, num_class]
        label2 = tf.one_hot(self.y2, task2_num_classes)
        losses1_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits1,
            labels=label1)

        losses2_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits2,
            labels=label2)

        self.loss_paired = tf.reduce_mean(losses1_arr) #+ tf.reduce_mean(losses2_arr)
