import tlm.dictionary.dict_reader_transformer as dr_transformer
import models.transformer.bert_common_v2 as bc
import tlm.model.base as base
import copy
import tensorflow as tf


def get_batch_and_seq_length(input_ids, expected_rank):
    input_shape = bc.get_shape_list(input_ids, expected_rank=expected_rank)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    return batch_size, seq_length


def get_pooler(sequence_output, config):
    with tf.compat.v1.variable_scope("pooler"):
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = tf.keras.layers.Dense(config.hidden_size,
                                                   activation=tf.keras.activations.tanh,
                                                   kernel_initializer=bc.create_initializer(config.initializer_range))(
            first_token_tensor)
    return pooled_output



class TransformerBase:
    def __init__(self, config, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
        self.config = config
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.batch_size, self.seq_length = get_batch_and_seq_length(input_ids, 2)
        self.initializer = base.create_initializer(config.initializer_range)
        self.key_dimension = config.key_dimension
        self.attention_mask = bc.create_attention_mask_from_input_mask(
            input_ids, self.input_mask)

    def forward_layer(self, prev_output):
        hidden_size = self.config.hidden_size
        layer_input = prev_output
        attention_output = bc.self_attention(layer_input,
                                             self.attention_mask,
                                             self.config,
                                             self.batch_size,
                                             self.seq_length,
                                             hidden_size,
                                             self.initializer)

        with tf.compat.v1.variable_scope("intermediate"):
            intermediate_output = bc.dense(self.config.intermediate_size, self.initializer,
                                           activation=bc.get_activation(self.config.hidden_act))(attention_output)

        with tf.compat.v1.variable_scope("output"):
            layer_output = bc.dense(hidden_size, self.initializer)(intermediate_output)
            layer_output = bc.dropout(layer_output, self.config.hidden_dropout_prob)
            layer_output = bc.layer_norm(layer_output + attention_output)
            prev_output = layer_output
        return intermediate_output, layer_output

    def get_embeddings(self, input_ids, segment_ids):
        config = self.config
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = bc.embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = bc.embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=segment_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
        return self.embedding_output


class MainTransformer(TransformerBase):
    def __init__(self, config, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
        super(MainTransformer, self).__init__(config, input_ids, input_mask, segment_ids, use_one_hot_embeddings)
        self.layers_before_key_pooling = 3 #
        self.all_layer_outputs = []
        self.key_dimension = config.key_dimension
        self.key_pooling = {
            "max_pooling": self.max_pooling,
            "mean_pooling": self.mean_pooling,
            "last_pooling": self.last_pooling,

        } [config.key_pooling_method ]

    def max_pooling(self, key_vectors):
        return tf.math.reduce_max(key_vectors, 1)

    def mean_pooling(self, key_vectors):
        num_tokens = tf.expand_dims(tf.cast(tf.reduce_sum(self.input_mask, axis=1), tf.float32), 1)
        key_sum = tf.reduce_sum(key_vectors, axis=1)  # [batch_size, mr_num_route]
        key_avg = tf.math.divide(key_sum, num_tokens)
        return key_avg

    def last_pooling(self, key_vectors):
        return key_vectors[:, -1, :]

    def build_key(self):
        with tf.compat.v1.variable_scope("embeddings"):
            input_tensor = self.get_embeddings(self.input_ids, self.segment_ids)

        with tf.compat.v1.variable_scope("encoder"):
            self.attention_mask = bc.create_attention_mask_from_input_mask(
                input_tensor, self.input_mask)
            prev_output = bc.reshape_to_matrix(input_tensor)
            for layer_idx in range(self.layers_before_key_pooling):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = self.forward_layer(prev_output)
                    self.all_layer_outputs.append(prev_output)

        with tf.variable_scope("mr_key"):
            key_vectors = tf.layers.dense(
                intermediate_output,
                self.key_dimension,
                kernel_initializer=self.initializer)
            key_vectors = bc.dropout(key_vectors, self.config.hidden_dropout_prob)
            key_vectors = tf.reshape(key_vectors, [self.batch_size, self.seq_length, -1])
            key_output = self.key_pooling(key_vectors)
        return key_output


class SecondTransformer(TransformerBase):
    def __init__(self, config, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
        super(SecondTransformer, self).__init__(config, input_ids, input_mask, segment_ids, use_one_hot_embeddings)
        self.all_layer_outputs = []
        self.key_dimension = config.key_dimension

    def build(self, key):
        return self.build_by_attetion(key)

    def build_by_attention(self, key):
        hidden_size = self.config.hidden_size
        with tf.compat.v1.variable_scope("embeddings"):
            input_tensor = self.get_embeddings(self.input_ids, self.segment_ids)

        with tf.compat.v1.variable_scope("encoder"):
            num_key_tokens = self.config.num_key_tokens
            project_dim = hidden_size * num_key_tokens
            raw_key = bc.dense(project_dim, self.initializer)(key)
            key_tokens = tf.reshape(raw_key, [self.batch_size, num_key_tokens, hidden_size])

            input_tensor = tf.concat([key_tokens, input_tensor], axis=1)

            mask_for_key = tf.ones_like([self.batch_size, num_key_tokens], dtype=tf.int32)
            self.input_mask = tf.concat([mask_for_key, self.input_mask], axis=1)
            self.seq_length = self.seq_length + num_key_tokens

            self.attention_mask = bc.create_attention_mask_from_input_mask(
                input_tensor, self.input_mask)
            prev_output = bc.reshape_to_matrix(input_tensor)
            for layer_idx in range(self.config.num_hidden_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = self.forward_layer(prev_output)
                    self.all_layer_outputs.append(prev_output)

            self.scores = bc.dense(1, self.initializer)(prev_output[:,0,:])

        return self.scores, prev_output


def align_keys(keys, ab_mapping):
    return tf.gather(keys, ab_mapping)

def select_value(a_size, ab_mapping, b_scores, b_items, method):
    # [b_size]


    b_size = bc.get_shape_list2(b_items)[0]
    indice = tf.stack([tf.range(b_size), ab_mapping], 1)

    collect_bin = tf.scatter_nd(indice, tf.ones([b_size]), [b_size, a_size])
    scattered_score = tf.transpose(tf.expand_dims(b_scores, 1) * collect_bin)
    # scattered_score :  [a_size, b_size], if not corresponding item, the score is zero

    if method == "max":
        selected_idx = tf.argmax(scattered_score, axis=1)

    elif method == "sample":
        remover = tf.transpose(tf.ones([b_size, a_size]) - collect_bin) * -10000.00
        scattered_score += remover
        samples = tf.random.categorical(scattered_score, 1)
        


    return tf.gather(b_items, selected_idx)




class SSDR(base.BertModelInterface):
    def __init__(self,
                 config,
                 d_config,
                 is_training,
                 input_ids,
                 input_mask,
                 token_type_ids,
                 d_input_ids,
                 d_input_mask,
                 d_segment_ids,
                 d_location_ids,
                 ab_mapping,
                 use_one_hot_embeddings=True,
                 scope=None,
                 ):
        super(SSDR, self).__init__()

        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        batch_size, seq_length = get_batch_and_seq_length(input_ids, 2)

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        if d_segment_ids is None:
            d_segment_ids = d_input_mask

        with tf.compat.v1.variable_scope(scope, default_name="bert"):
            with tf.compat.v1.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = bc.embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = bc.embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.compat.v1.variable_scope("dict"):
                with tf.compat.v1.variable_scope("embeddings"):
                    (self.d_embedding_output, self.d_embedding_table) = bc.embedding_lookup(
                        input_ids=d_input_ids,
                        vocab_size=config.vocab_size,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name="word_embeddings",
                        use_one_hot_embeddings=use_one_hot_embeddings)

                    self.d_embedding_output = bc.embedding_postprocessor(
                        input_tensor=self.d_embedding_output,
                        use_token_type=True,
                        token_type_ids=d_segment_ids,
                        token_type_vocab_size=config.type_vocab_size,
                        token_type_embedding_name="token_type_embeddings",
                        use_position_embeddings=True,
                        position_embedding_name="position_embeddings",
                        initializer_range=config.initializer_range,
                        max_position_embeddings=config.max_position_embeddings,
                        dropout_prob=config.hidden_dropout_prob)

            with tf.compat.v1.variable_scope("encoder"):
                self.main_transformer = MainTransformer(
                    config, input_ids, input_mask, token_type_ids, use_one_hot_embeddings)

                key_out = self.main_transformer.build_key()

                self.dict_tranformer = SecondTransformer(
                    d_config, d_input_ids, d_input_mask, d_segment_ids, use_one_hot_embeddings)


                aligned_key = tf.gather(key_out, ab_mapping)

                scores, last_layers = self.dict_tranformer.build(aligned_key)

                select_value(ab_mapping, scores, last_layers)

                all_encoder_layers = self.main_transformer.build_remain(value_out)

                self.all_encoder_layers = all_encoder_layers
                self.sequence_output = self.all_encoder_layers[-1]
                self.pooled_output = get_pooler(self.sequence_output, config)

                self.dict_sequence_output = self.dict_layers[-1]
