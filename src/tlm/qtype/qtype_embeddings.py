import tensorflow as tf

from models.transformer.bert_common_v2 import create_initializer


class QTypeEmbedding(tf.keras.layers.Layer):
    def __init__(self, config, inner_layer):
        super(QTypeEmbedding, self).__init__()
        self.config = config
        self.qtype_len = config.qtype_len
        self.hidden_size = config.hidden_size
        self.inner_layer: tf.keras.layers.Layer = inner_layer
        self.embedding_table = None
        config = self.config
        embedding_size = config.hidden_size * self.qtype_len
        self.embedding_table = self.add_weight(
            shape=(config.q_voca_size, embedding_size),
            initializer=create_initializer(config.initializer_range),
            name="qtype_modeling/qtype_embedding",
            trainable=True
        )

    def reshape_embedding_out(self, embedding_raw):
        return tf.reshape(embedding_raw, [-1, self.qtype_len, self.hidden_size])

    def get_qtype_embedding_table(self):
        return self.embedding_table

    def get_inner_layer(self):
        return self.inner_layer


#  https://gist.github.com/ericjang/1001afd374c2c3b7752545ce6d9ed349
def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, is_training):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    if is_training:
        y = logits + sample_gumbel(tf.shape(logits))
    else:
        y = logits
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False, is_training=True):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, is_training)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def decay_fn(start_value, rate, step, decay_steps):
    return start_value * tf.pow(rate,  (step / decay_steps))


class QTypeEmbeddingWeightPred(QTypeEmbedding):
    def __init__(self, config, inner_layer, is_training):
        super(QTypeEmbeddingWeightPred, self).__init__(config, inner_layer)
        self.temperature = None
        try:
            init_temperature = tf.constant(1.0)
            current_step = tf.compat.v1.train.get_or_create_global_step()
            temperature = decay_fn(init_temperature, 0.1, current_step, self.config.train_steps)
        except AttributeError:
            temperature = tf.constant(1.0)
        self.weight_projection = tf.keras.layers.Dense(config.q_voca_size)
        self.temperature = temperature
        self.is_training = is_training

    def call(self, inputs, **kwargs):
        raw_rep = inputs
        pred_type_emb = self.get_inner_layer()(raw_rep)
        qtype_logits = self.weight_projection(pred_type_emb)
        self.qtype_logits = qtype_logits
        qtype_weights = gumbel_softmax(qtype_logits, self.temperature, False, self.is_training)
        self.qtype_weights = qtype_weights
        qtype_embeddings = tf.matmul(qtype_weights, self.embedding_table)
        return self.reshape_embedding_out(qtype_embeddings)

    def get_qtype_raw_logits(self):
        return self.qtype_logits

    def get_qtype_weights(self):
        return self.qtype_weights


class QWordPredict(QTypeEmbedding):
    def __init__(self, config, inner_layer, is_training):
        super(QWordPredict, self).__init__(config, inner_layer)
        self.weight_projection = tf.keras.layers.Dense(config.q_voca_size)
        self.temperature = tf.constant(1.0)
        self.is_training = is_training

    def call(self, inputs, **kwargs):
        raw_rep = inputs
        pred_type_emb = self.get_inner_layer()(raw_rep)
        qtype_logits = self.weight_projection(pred_type_emb)
        self.qtype_logits = qtype_logits
        qtype_weights = gumbel_softmax(qtype_logits, self.temperature,
                                       is_training=self.is_training)
        self.qtype_weights = qtype_weights
        return self.qtype_weights

    def get_qtype_raw_logits(self):
        return self.qtype_logits

    def get_qtype_weights(self):
        return self.qtype_weights


@tf.custom_gradient
def nn_table_lookup(v, table):
    max_idx = get_closest_from_table(v, table)
    output = tf.nn.embedding_lookup(params=table, ids=max_idx)  # [batch_size, hidden_dim]

    def custom_grad(dy):
        grad = dy, tf.zeros_like(table)  # compute gradient
        return grad
    return output, custom_grad


def get_closest_from_table(v, table):
    v_ex = tf.expand_dims(v, 1)  # [batch_size, 1, hidden_dim]
    table_ex = tf.expand_dims(table, 0)  # [1, voca_size, hidden_dim]
    scores = tf.matmul(v_ex, table_ex, transpose_b=True)  # [batch_size, 1, voca_size]
    max_idx = tf.reshape(tf.argmax(scores, axis=2), [-1])
    return max_idx


# Option 1-1. Feed directly, no table
class QTypeEmbeddingEmbPredDirect(QTypeEmbedding):
    def __init__(self, config, inner_layer):
        super(QTypeEmbeddingEmbPredDirect, self).__init__(config, inner_layer)

    def call(self, inputs, **args):
        raw_rep = inputs
        pred_type_emb = self.get_inner_layer()(raw_rep)
        return self.reshape_embedding_out(pred_type_emb)


class QTypeEmbeddingEmbPredDebug(QTypeEmbedding):
    def __init__(self, config):
        super(QTypeEmbeddingEmbPredDebug, self).__init__(config)

    def call(self, inputs, **args):
        return self.reshape_embedding_out(inputs[:, 1, 0, :768])


# 1-3 VQ-VAE (Oord et al)
class QTypeEmbeddingEmbPred(QTypeEmbedding):
    def __init__(self, config, inner_layer):
        super(QTypeEmbeddingEmbPred, self).__init__(config, inner_layer)

    def call(self, inputs, **args):
        raw_rep = inputs
        pred_type_emb = self.get_inner_layer()(raw_rep)
        table = self.get_qtype_embedding_table()
        self.pred_type_emb = pred_type_emb

        max_idx = get_closest_from_table(pred_type_emb, table)
        qtype_embeddings = tf.nn.embedding_lookup(params=table, ids=max_idx)  # [batch_size, hidden_dim]

        # qtype_embeddings = nn_table_lookup(pred_type_emb, table)

        loss_a = tf.nn.l2_loss(tf.stop_gradient(pred_type_emb) - qtype_embeddings)
        loss_b = tf.nn.l2_loss(pred_type_emb - tf.stop_gradient(qtype_embeddings))

        self.add_loss(loss_a)
        self.add_loss(loss_b)
        return self.reshape_embedding_out(qtype_embeddings)

    def get_qtype_weights(self):
        v_ex = tf.expand_dims(self.pred_type_emb, 1)  # [batch_size, 1, hidden_dim]
        table_ex = tf.expand_dims(self.get_qtype_embedding_table(), 0)  # [1, voca_size, hidden_dim]
        scores = tf.matmul(v_ex, table_ex, transpose_b=True)  # [batch_size, 1, voca_size]
        return scores

    def get_qtype_raw_logits(self):
        return self.get_qtype_weights()
