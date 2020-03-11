import tensorflow as tf

import data_generator.NLI.nli_info
from models.transformer import bert
from task.transformer_est import Classification


class transformer_concat:
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

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
        if method in [0, 1, 3, 4, 5, 6]:
            self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        elif method in [2]:
            self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        use_one_hot_embeddings = use_tpu
        with tf.variable_scope("part1"):
            self.model1 = bert.BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)

        with tf.variable_scope("part2"):
            self.model2 = bert.BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)

        enc = tf.concat([self.model1.get_sequence_output(),
                   self.model2.get_sequence_output()], axis=2)

        pred, loss = task.predict(enc, label_ids, True)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc