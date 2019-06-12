from task.transformer_est import Transformer, Classification
from models.transformer import bert
import tensorflow as tf
from data_generator.NLI import nli

class transformer_distribution:
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
        label_ids = tf.placeholder(tf.float32, [None, 3])

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

        feature = self.model.get_pooled_output()


        def dense_softmax(feature, name):
            logits = tf.layers.dense(feature, 2, name=name)
            sout = tf.nn.softmax(logits)
            return sout

        alpha = dense_softmax(feature, "dense_alpha")  # Probability of being Argument P(Arg)
        beta = dense_softmax(feature, "dense_beta")    # P(Arg+|Arg)
        gamma = dense_softmax(feature, "dense_gamma")  # P(not Noise)
        self.alpha = alpha[:, 0]
        self.beta = beta[:, 0]
        self.gamma = gamma[:, 0]

        p1_prior = 0.2
        p2_prior = 0.2
        p0_prior = 1 - p1_prior - p2_prior

        p1 = alpha[:, 0] * beta[:, 0] * gamma[:, 0] + gamma[:, 1] * p1_prior
        p2 = alpha[:, 0] * beta[:, 1] * gamma[:, 0] + gamma[:, 1] * p2_prior
        p0 = alpha[:, 1] * gamma[:, 0] + gamma[:, 1] * p0_prior


        pred = tf.stack([p0,p1,p2], axis=1)
        log_likelihood = tf.log(pred) * label_ids
        loss = - tf.reduce_mean(log_likelihood)
        self.pred = pred
        self.loss = loss
