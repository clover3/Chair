import tensorflow as tf

from models.transformer import bert_common_v2 as bc


class SelfAttentionLayer:
    def __init__(self, config):
        hidden_size = config.hidden_size
        initializer = bc.create_initializer(config.initializer_range)

        attention_head_size = int(hidden_size / config.num_attention_heads)
        self.attention_head_size = attention_head_size
        num_attention_heads = config.num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_dropout_prob = config.hidden_dropout_prob

        with tf.compat.v1.variable_scope("attention"):
            with tf.compat.v1.variable_scope("self"):
                self.query_layer = tf.keras.layers.Dense(
                    num_attention_heads * attention_head_size,
                    activation=None,
                    name="query",
                    kernel_initializer=initializer)

                self.key_layer = tf.keras.layers.Dense(
                    num_attention_heads * attention_head_size,
                    activation=None,
                    name="key",
                    kernel_initializer=initializer)
                self.value_layer = tf.keras.layers.Dense(
                    num_attention_heads * attention_head_size,
                    activation=None,
                    name="value",
                    kernel_initializer=initializer)
                with tf.compat.v1.variable_scope("output"):
                    self.output_layer = tf.keras.layers.Dense(config.hidden_size,
                                                 kernel_initializer=initializer,
                                                 )

    def call(self, layer_input, attention_mask, batch_size, seq_length):
        attention_heads = []
        with tf.compat.v1.variable_scope("attention"):
            with tf.compat.v1.variable_scope("self"):
                attention_head = bc.attention_layer2(
                    from_tensor=layer_input,
                    to_tensor=layer_input,
                    query_ff=self.query_layer,
                    key_ff=self.key_layer,
                    value_ff=self.value_layer,
                    attention_mask=attention_mask,
                    num_attention_heads=self.num_attention_heads,
                    size_per_head=self.attention_head_size,
                    attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                    do_return_2d_tensor=True,
                    batch_size=batch_size,
                    from_seq_length=seq_length,
                    to_seq_length=seq_length)
                attention_heads.append(attention_head)

            attention_output = None
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                # In the case where we have other sequences, we just concatenate
                # them to the self-attention head before the projection.
                attention_output = tf.concat(attention_heads, axis=-1)

            # Run a linear projection of `hidden_size` then add a residual
            # with `layer_input`.
            with tf.compat.v1.variable_scope("output"):
                attention_output = self.output_layer(attention_output)
                attention_output = bc.dropout(attention_output, self.hidden_dropout_prob)
                attention_output = bc.layer_norm(attention_output + layer_input)
        return attention_output


class ForwardLayer:
    # layer knows nothing about hierarchical structure.
    def __init__(self, config, initializer):
        self.config = config
        self.self_attention = SelfAttentionLayer(config)
        with tf.compat.v1.variable_scope("intermediate"):
            self.intermediate_ff = bc.dense(self.config.intermediate_size, initializer,
                                           activation=bc.get_activation(self.config.hidden_act))
        with tf.compat.v1.variable_scope("output"):
            self.output_ff = bc.dense(config.hidden_size, initializer)

    def apply(self, prev_output, batch_size, seq_length, attention_mask):
        layer_input = prev_output
        attention_output = self.self_attention.call(layer_input,
                                                    attention_mask,
                                                    batch_size,
                                                    seq_length,
                                                    )
        with tf.compat.v1.variable_scope("intermediate"):
            intermediate_output = self.intermediate_ff(attention_output)
        with tf.compat.v1.variable_scope("output"):
            layer_output = self.output_ff(intermediate_output)
            layer_output = bc.dropout(layer_output, self.config.hidden_dropout_prob)
            layer_output = bc.layer_norm(layer_output + attention_output)
        return intermediate_output, layer_output

    def apply_3d(self, input_tensor, batch_size, seq_length, attention_mask):
        input_shape = bc.get_shape_list2(input_tensor)
        input_tensor = bc.reshape_to_matrix(input_tensor)
        intermediate_output, layer_output = self.apply(input_tensor, batch_size, seq_length, attention_mask)

        return bc.reshape_from_matrix2(layer_output, input_shape)


class Embedding:
    def __init__(self, config, use_one_hot_embeddings):
        self.config = config
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_output = None
        self.embedding_table = None

    def apply(self, input_ids, segment_ids):
        config = self.config
        initializer = bc.create_initializer(config.initializer_range)
        self.embedding_table = tf.compat.v1.get_variable(
            name="word_embeddings",
            shape=[config.vocab_size, config.hidden_size],
            initializer=initializer)
        self.token_type_table = tf.compat.v1.get_variable(
                name="token_type_embeddings",
                shape=[config.type_vocab_size, config.hidden_size],
                initializer=initializer)
        self.full_position_embeddings = tf.compat.v1.get_variable(
            name="position_embeddings",
            shape=[config.max_position_embeddings, config.hidden_size],
            initializer=initializer)

        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = bc.embedding_lookup2(
            input_ids=input_ids,
            embedding_table=self.embedding_table,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = bc.embedding_postprocessor2(
            input_tensor=self.embedding_output,
            token_type_table=self.token_type_table,
            full_position_embeddings=self.full_position_embeddings,
            use_token_type=True,
            token_type_ids=segment_ids,
            token_type_vocab_size=config.type_vocab_size,
            use_position_embeddings=True,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
        return self.embedding_output


class Embedding2:
    def __init__(self):
        self.embedding_output = None
        self.embedding_table = None

    def apply(self, input_ids, segment_ids,
              initializer_range, vocab_size, hidden_size, type_vocab_size, max_position_embeddings,
              hidden_dropout_prob, use_one_hot_embeddings):
        initializer = bc.create_initializer(initializer_range)
        self.embedding_table = tf.compat.v1.get_variable(
            name="word_embeddings",
            shape=[vocab_size, hidden_size],
            initializer=initializer)
        self.token_type_table = tf.compat.v1.get_variable(
            name="token_type_embeddings",
            shape=[type_vocab_size, hidden_size],
            initializer=initializer)
        self.full_position_embeddings = tf.compat.v1.get_variable(
            name="position_embeddings",
            shape=[max_position_embeddings, hidden_size],
            initializer=initializer)

        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = bc.embedding_lookup2(
            input_ids=input_ids,
            embedding_table=self.embedding_table,
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = bc.embedding_postprocessor2(
            input_tensor=self.embedding_output,
            token_type_table=self.token_type_table,
            full_position_embeddings=self.full_position_embeddings,
            use_token_type=True,
            token_type_ids=segment_ids,
            token_type_vocab_size=type_vocab_size,
            use_position_embeddings=True,
            max_position_embeddings=max_position_embeddings,
            dropout_prob=hidden_dropout_prob)
        return self.embedding_output
