

from models.transformer.bert import *

class ScoreCombinerFF:
    def __init__(self, hp):
        seq_length = hp.seq_max
        input_tensor = tf.placeholder(tf.float32, [None, seq_length, hp.hidden_units])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        self.x_list = [input_tensor, input_mask, segment_ids]
        hidden_dropout_prob = 0.1
        data_size = seq_length * hp.hidden_units
        in_tensor = tf.reshape(input_tensor, [-1, data_size])

        for i in range(3):
            output = tf.layers.dense(in_tensor, data_size, activation=tf.nn.relu)
            output = dropout(output, hidden_dropout_prob)
            in_tensor = layer_norm(output + in_tensor)

        in_tensor = tf.layers.dense(in_tensor, hp.hidden_units, activation=tf.nn.relu)
        for i in range(2):
            output = tf.layers.dense(in_tensor, hp.hidden_units, activation=tf.nn.relu)
            output = dropout(output, hidden_dropout_prob)
            in_tensor = layer_norm(output + in_tensor)

        self.logits = tf.layers.dense(in_tensor, 1)  # [None, hp.hidden_units]
        paired = tf.reshape(self.logits, [-1, 2])
        losses = tf.maximum(hp.alpha - (paired[:, 1] - paired[:, 0]), 0)
        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', self.loss)


class ScoreCombinerLSTM:
    def __init__(self, hp):
        seq_length = hp.seq_max
        input_tensor = tf.placeholder(tf.float32, [None, seq_length, hp.hidden_units])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        self.x_list = [input_tensor, input_mask, segment_ids]
        hidden_dropout_prob = 0.1
        data_size = seq_length * hp.hidden_units

        in_tensor = NotImplemented

        self.logits = tf.layers.dense(in_tensor, 1)  # [None, hp.hidden_units]
        paired = tf.reshape(self.logits, [-1, 2])
        losses = tf.maximum(hp.alpha - (paired[:, 1] - paired[:, 0]), 0)
        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', self.loss)




class ScoreCombinerF1:
    def __init__(self, hp):
        seq_length = hp.seq_max
        input_tensor = tf.placeholder(tf.float32, [None, seq_length, hp.hidden_units])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        self.x_list = [input_tensor, input_mask, segment_ids]
        hidden_dropout_prob = 0.1
        data_size = seq_length * hp.hidden_units
        t0 = tf.reduce_sum(input_tensor, axis=2)

        in_tensor = tf.layers.dense(t0, seq_length, activation=tf.nn.relu)
        for i in range(2):
            output = tf.layers.dense(in_tensor, seq_length, activation=tf.nn.relu)
            output = dropout(output, hidden_dropout_prob)
            in_tensor = layer_norm(output + in_tensor)

        self.logits = tf.layers.dense(in_tensor, 1)  # [None, hp.hidden_units]
        paired = tf.reshape(self.logits, [-1, 2])
        losses = tf.maximum(hp.alpha - (paired[:, 1] - paired[:, 0]), 0)
        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', self.loss)



class ScoreCombinerMax:
    def __init__(self, hp):
        seq_length = hp.seq_max
        input_tensor = tf.placeholder(tf.float32, [None, seq_length, hp.hidden_units])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        self.x_list = [input_tensor, input_mask, segment_ids]
        hidden_dropout_prob = 0.1
        W = tf.Variable(tf.ones((seq_length, hp.hidden_units), tf.float32))

        #t0 = tf.layers.dense(input_tensor, hp.hidden_units)
        t0 = input_tensor * W
        t1 = tf.reduce_max(t0, axis=1)
        t2 = tf.reduce_sum(t1, axis=1)

        self.logits = t2
        paired = tf.reshape(self.logits, [-1, 2])
        losses = tf.maximum(hp.alpha - (paired[:, 1] - paired[:, 0]), 0)
        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', self.loss)



class ScoreCombiner:
    def __init__(self, hp):
        seq_length = hp.seq_max
        input_tensor = tf.placeholder(tf.float32, [None, seq_length, hp.hidden_units])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        self.x_list = [input_tensor, input_mask, segment_ids]
        input_tensor = tf.contrib.layers.layer_norm(input_tensor)
        initializer_range = 1
        with tf.variable_scope("encoder"):
            # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
            # mask of shape [batch_size, seq_length, seq_length] which is used
            # for the attention scores.

            self.embedding_output = embedding_postprocessor(
                input_tensor=input_tensor,
                use_token_type=True,
                token_type_ids=segment_ids,
                token_type_vocab_size=2,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=initializer_range,
                max_position_embeddings=seq_length,
                dropout_prob=0.1)

            attention_mask = create_attention_mask_from_input_mask(
                input_tensor, input_mask)

            self.all_encoder_layers = transformer_model(
                input_tensor=input_tensor,
                attention_mask=attention_mask,
                hidden_size=hp.hidden_units,
                num_hidden_layers=hp.num_blocks,
                num_attention_heads=hp.num_heads,
                intermediate_size=hp.intermediate_size,
                intermediate_act_fn=get_activation('gelu'),
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                initializer_range=initializer_range,
                do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
            self.pooled_output = tf.layers.dense(
                first_token_tensor,
                hp.hidden_units,
                activation=tf.tanh,
                kernel_initializer=create_initializer(initializer_range))

            self.logits = tf.layers.dense(self.pooled_output, 1, name="reg_dense")  # [ None, 1]

            paired = tf.reshape(self.logits, [-1, 2])
            losses = tf.maximum(hp.alpha - (paired[:, 1] - paired[:, 0]), 0)
            self.loss = tf.reduce_mean(losses)
            tf.summary.scalar('loss', self.loss)

