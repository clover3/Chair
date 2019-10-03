from task.transformer_est import Transformer, Classification
from models.transformer import bert
import tensorflow as tf
from data_generator.NLI import nli
from trainer import tf_module

class transformer_ql:
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
        task = Classification(nli.num_classes)

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        scores = tf.placeholder(tf.float32, [None])


        self.query = tf.placeholder(tf.int64, [None, hp.query_seq_len])
        self.q_mask = tf.placeholder(tf.int64, [None,hp.query_seq_len])

        self.q_mask_f = tf.cast(self.q_mask, tf.float32)
#        self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])

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

        input_tensor = self.model.get_sequence_output()

        def get_query_encode(query):
            flat_input_ids = tf.reshape(query, [-1])
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=voca_size)
            output = tf.matmul(one_hot_input_ids, self.model.get_embedding_table())
            input_shape = bert.get_shape_list(query)
            embedding_size = hp.hidden_units
            new_size = input_shape + [ embedding_size]
            print(input_shape)
            print(new_size)
            output = tf.reshape(output, new_size)
            return output

        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                token_enc = tf.layers.dense(
                    input_tensor,
                    units=hp.hidden_units,
                    activation=bert.get_activation(config.hidden_act),
                    kernel_initializer=bert.create_initializer(
                        config.initializer_range))
                token_enc = bert.layer_norm(token_enc)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[voca_size],
                initializer=tf.zeros_initializer())

            def expand_transpose(t):
                return tf.transpose(tf.expand_dims(t, 0), perm=[0,2,1])

            emb_ex = expand_transpose(self.model.get_embedding_table())

            logits = tf.tensordot(token_enc, emb_ex, [[2],[1]])
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            self.log_probs = log_probs

            query_vector = get_query_encode(self.query)
            query_likelihood_i = tf.matmul(token_enc, query_vector, transpose_b=True) # [None, seq_len, H] * [None, H, Q_len] -> [None, sqe_len, Q_len]
            query_likelihood_i = tf.reduce_sum(query_likelihood_i * tf.expand_dims(self.q_mask_f, 1), axis=2) #[None, seq_len]

            input_mask_f = tf.cast(input_mask, tf.float32)
            query_likelihood_i = query_likelihood_i
            #query_likelihood = tf.reduce_sum(query_likelihood_i, axis=1) / tf.reduce_sum(input_mask_f)
            query_likelihood = tf.reduce_max(query_likelihood_i, axis=1)
            self.ql_score = query_likelihood
