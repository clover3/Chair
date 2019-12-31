import os

import tensorflow as tf

import models.transformer.bert_common_v2 as bc
from tlm.model import base
from tlm.model_cnfig import JsonConfig


def get_per_layer_config(all_config:JsonConfig):
    keys = ["attention_probs_dropout_prob", "hidden_act", "hidden_dropout_prob", "hidden_size",
              "initializer_range", "intermediate_size", "num_attention_heads"]

    d = {}
    for key in keys:
        d[key] = all_config.__dict__[key]

    return JsonConfig.from_dict(d)


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


class LowerTransformer(tf.keras.layers.Layer):
    def __init__(self, config, n_layers, use_one_hot_embeddings, **kwargs):
        kwargs['autocast'] = False
        super(LowerTransformer, self).__init__(kwargs)
        self.n_layers = n_layers
        self.all_layer_outputs = []
        self.last_key_layer = None
        self.config = config
        self.embedding = None
        self.layer_list = []
        self.initializer = base.create_initializer(config.initializer_range)
        self.attention_mask = None
        self.use_one_hot_embeddings = use_one_hot_embeddings
        with tf.compat.v1.variable_scope("encoder"):
            for layer_idx in range(self.n_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    layer = ForwardLayer(self.config, self.initializer)
                    self.layer_list.append(layer)

    def call(self, input_ids, input_mask, segment_ids):
        with tf.compat.v1.variable_scope("embeddings"):
            self.embedding_layer = Embedding(self.config, self.use_one_hot_embeddings)
            input_tensor = self.embedding_layer.apply(input_ids, segment_ids)
            self.embedding_output = input_tensor
            input_shape = bc.get_shape_list2(input_tensor)
            batch_size, seq_length, _ = input_shape
        with tf.compat.v1.variable_scope("lower"):
            self.attention_mask = bc.create_attention_mask_from_input_mask2(
                input_tensor, input_mask)
            prev_output = bc.reshape_to_matrix(input_tensor)
            for layer_idx in range(self.n_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    layer = self.layer_list[layer_idx]
                    intermediate_output, prev_output = layer.apply(prev_output, batch_size, seq_length,
                                                             self.attention_mask)
                    final_output = bc.reshape_from_matrix2(prev_output, input_shape)
                    self.all_layer_outputs.append(final_output)

        return prev_output


class UpperTransformer(tf.keras.layers.Layer):
    def __init__(self, config, n_layers, **kwargs):
        kwargs['autocast'] = False
        super(UpperTransformer, self).__init__(kwargs)
        self.n_layers = n_layers
        self.all_layer_outputs = []
        self.last_key_layer = None
        self.config = config
        self.embedding = None
        self.layer_list = []
        self.initializer = base.create_initializer(config.initializer_range)

        for layer_idx in range(self.n_layers):
            with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                layer = ForwardLayer(self.config, self.initializer)
                self.layer_list.append(layer)

    def call(self, input_vectors, attention_mask):
        prev_output = input_vectors
        input_shape = bc.get_shape_list2(input_vectors)
        batch_size, seq_length, _ = input_shape
        prev_output = bc.reshape_to_matrix(prev_output)
        for layer_idx in range(self.n_layers):
            with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                layer = self.layer_list[layer_idx]
                intermediate_output, prev_output = layer.apply(prev_output, batch_size, seq_length,
                                                               attention_mask)
                final_output = bc.reshape_from_matrix2(prev_output, input_shape)
                self.all_layer_outputs.append(final_output)

        return prev_output

    def get_last_layer_output(self):
        return self.all_layer_outputs[-1]

def split_and_append_sep(input_ids,
                input_mask,
                segment_ids,
               seq_length:int,
               window_length:int,
               CLS_ID,
               EOW_ID):
    special_tokens = 2  # CLS, SEP
    src_window_length = window_length - special_tokens
    num_window = int(seq_length / src_window_length)

    window_input_ids_list = []
    window_input_mask_list = []
    window_segment_ids_list = []
    for window_idx in range(num_window):
        st = window_idx * src_window_length
        ed = (window_idx+1) * src_window_length
        window_input_ids_list.append(input_ids[:, st:ed])
        window_input_mask_list.append(input_mask[:, st:ed])
        window_segment_ids_list.append(segment_ids[:, st:ed])


    stacked_input_ids = tf.stack(window_input_ids_list, 1) # [batch_size, num_window, src_window_length]
    stacked_input_mask= tf.stack(window_input_mask_list, 1)  # [batch_size, num_window, src_window_length]
    stacked_segment_ids = tf.stack(window_segment_ids_list, 1)  # [batch_size, num_window, src_window_length]

    batch_size, num_window, _ = bc.get_shape_list2(stacked_input_ids)
    edge_shape = [batch_size, num_window, 1]
    cls_arr = tf.ones(edge_shape, tf.int32) * CLS_ID
    eow_arr = tf.ones(edge_shape, tf.int32) * EOW_ID

    stacked_input_ids = tf.concat([cls_arr, stacked_input_ids, eow_arr], axis=2)

    mask_edge = tf.ones(edge_shape, tf.int32)
    stacked_input_mask = tf.concat([mask_edge, stacked_input_mask, mask_edge], axis=2)

    edge1 = stacked_segment_ids[:, :, 0:1]
    edge2 = stacked_segment_ids[:, :, -2:-1]
    stacked_segment_ids = tf.concat([edge1, stacked_segment_ids, edge2], axis=2)

    return stacked_input_ids, stacked_input_mask, stacked_segment_ids


class SeroAlpha(base.BertModelInterface):
    def __init__(self,
                 config, # This is different from BERT config,
                 is_training,
                 use_one_hot_embeddings=True,
                 ):
        super(SeroAlpha, self).__init__()

        self.total_sequence_length = config.total_sequence_length
        self.window_size = config.window_size

        self.lower_module = LowerTransformer(config, config.lower_layers, use_one_hot_embeddings)
        if config.compare_attrib_value_safe_lambda("", lambda x: x > 0):
            self.mid_layers = MidModuleExpanding(config)
        else:
            self.mid_layers = MidModule(config)
        self.upper_module = UpperTransformer(config)
        self.skip_mid_layer = config.compare_attrib_value_safe("skip_mid_layer", True)

    def call(self, stacked_input_ids, stacked_input_mask,
                 stacked_segment_ids, use_context):
        self.lower_module.call(stacked_input_ids,
                                             stacked_input_mask,
                                             stacked_segment_ids,
                                             )

        lower_module_last_layer = self.lower_module.all_layer_outputs[-1]
        if not self.skip_mid_layer:
            window_vectors = lower_module_last_layer [:, -1, :]
            context_vectors = self.mid_layers.call(window_vectors, use_context) # [num_window, hidden_size]
            context_vectors = tf.expand_dims(context_vectors, 1)
            input_vectors = tf.concat([lower_module_last_layer [:, :-1, :], context_vectors], axis=1)
        else:
            input_vectors = lower_module_last_layer

        with tf.compat.v1.variable_scope("upper"):
            self.upper_module.call(input_vectors, self.lower_module.attention_mask)
        self.embedding_table = self.lower_module.embedding_layer.embedding_table
        self.sequence_output = self.upper_module.all_layer_outputs[-1]
        self.all_encoder_layers = self.lower_module.all_layer_outputs + self.upper_module.all_layer_outputs
        self.embedding_output = self.lower_module.embedding_output
        return self.sequence_output


class SeroBeta(base.BertModelInterface):
    def __init__(self,
                 config, # This is different from BERT config,
                 is_training,
                 use_one_hot_embeddings=True,
                 ):
        super(SeroBeta, self).__init__()

        if not is_training:
            config.set_attrib("hidden_dropout_prob", 0.0)
            config.set_attrib("attention_probs_dropout_prob", 0.0)

        self.total_sequence_length = config.total_sequence_length
        self.window_size = config.window_size

        self.lower_module = LowerTransformer(config, config.lower_layers, use_one_hot_embeddings)
        self.mid_layers = MidModule(config)
        self.upper_module = UpperTransformer(config)
        self.skip_mid_layer = config.compare_attrib_value_safe("skip_mid_layer", True)

    def call(self, stacked_input_ids, stacked_input_mask,
                 stacked_segment_ids, use_context):
        self.lower_module.call(stacked_input_ids,
                               stacked_input_mask,
                               stacked_segment_ids,
                               )

        lower_module_last_layer = self.lower_module.all_layer_outputs[-1]
        window_vectors = lower_module_last_layer [:, -1, :]
        self.mid_layers.call(window_vectors, use_context) # [num_window, n, hidden_size ]
        context_vectors = self.mid_layers.concat_all_layers()
        input_vectors = tf.concat([lower_module_last_layer, context_vectors], axis=1)

        added_tokens = self.mid_layers.n_layers
        attention_mask = tf.pad(self.lower_module.attention_mask,
                                [[0,0], [0, added_tokens], [0, added_tokens]], 'CONSTANT', constant_values=1)

        with tf.compat.v1.variable_scope("upper"):
            self.upper_module.call(input_vectors, attention_mask)
        self.embedding_table = self.lower_module.embedding_layer.embedding_table
        self.sequence_output = self.upper_module.all_layer_outputs[-1]
        self.all_encoder_layers = self.lower_module.all_layer_outputs + self.upper_module.all_layer_outputs
        self.embedding_output = self.lower_module.embedding_output
        return self.sequence_output



def r3to2(t):
    a, b, c = bc.get_shape_list2(t)
    return tf.reshape(t, [-1, c])



class SeroGamma(base.BertModelInterface):
    def __init__(self,
                 config, # This is different from BERT config,
                 is_training,
                 use_one_hot_embeddings=True,
                 ):
        super(SeroGamma, self).__init__()

        if not is_training:
            config.set_attrib("hidden_dropout_prob", 0.0)
            config.set_attrib("attention_probs_dropout_prob", 0.0)

        self.total_sequence_length = config.total_sequence_length
        self.window_size = config.window_size

        self.lower_module = LowerTransformer(config, config.lower_layers, use_one_hot_embeddings)
        self.mid_layers = MidModuleGamma(config)
        self.upper_module = UpperTransformer(config)
        self.skip_mid_layer = config.compare_attrib_value_safe("skip_mid_layer", True)

    def call(self, stacked_input_ids, stacked_input_mask,
                 stacked_segment_ids, use_context):
        self.lower_module.call(r3to2(stacked_input_ids),
                               r3to2(stacked_input_mask),
                               r3to2(stacked_segment_ids),
                               )

        lower_module_last_layer = self.lower_module.all_layer_outputs[-1] # [ batch_size * num_window, seq_length, hidden_size)
        window_vectors = lower_module_last_layer [:, -1, :]

        batch_size, num_window, seq_length = bc.get_shape_list2(stacked_input_ids)
        window_vectors = tf.reshape(window_vectors, [batch_size, num_window, -1])
        print("window_vectors", window_vectors.shape)
        context_vectors = self.mid_layers.call(window_vectors, use_context) # [batch_size, num_window, hidden_size ]
        print("context_vectors ", context_vectors.shape)
        context_vectors = tf.reshape(context_vectors, [batch_size * num_window, 1, -1])
        input_vectors = tf.concat([lower_module_last_layer, context_vectors], axis=1)

        added_tokens = 1
        attention_mask = tf.pad(self.lower_module.attention_mask,
                                [[0,0], [0, added_tokens], [0, added_tokens]], 'CONSTANT', constant_values=1)

        with tf.compat.v1.variable_scope("upper"):
            self.upper_module.call(input_vectors, attention_mask)
        self.embedding_table = self.lower_module.embedding_layer.embedding_table
        self.sequence_output = self.upper_module.all_layer_outputs[-1]
        self.all_encoder_layers = self.lower_module.all_layer_outputs + self.upper_module.all_layer_outputs

        self.all_encoder_layers = []
        self.embedding_output = self.lower_module.embedding_output
        return self.sequence_output


class SeroDelta(base.BertModelInterface):
    def __init__(self,
                 config,  # This is different from BERT config,
                 is_training,
                 use_one_hot_embeddings=True,
                 ):
        super(SeroDelta, self).__init__()

        if not is_training:
            config.set_attrib("hidden_dropout_prob", 0.0)
            config.set_attrib("attention_probs_dropout_prob", 0.0)

        self.total_sequence_length = config.total_sequence_length
        self.window_size = config.window_size

        self.lower_module = LowerTransformer(config, config.lower_layers, use_one_hot_embeddings)

        per_layer_config = get_per_layer_config(config)
        self.mid_layers = UpperTransformer(per_layer_config, config.middle_layers)
        self.upper_module = UpperTransformer(per_layer_config, config.upper_layers)
        self.skip_mid_layer = config.compare_attrib_value_safe("skip_mid_layer", True)

    def call(self, stacked_input_ids, stacked_input_mask,
             stacked_segment_ids, use_context):
        self.lower_module.call(r3to2(stacked_input_ids),
                               r3to2(stacked_input_mask),
                               r3to2(stacked_segment_ids),
                               )

        lower_module_last_layer = self.lower_module.all_layer_outputs[
            -1]  # [ batch_size * num_window, seq_length, hidden_size)
        window_vectors = lower_module_last_layer[:, -1, :]

        batch_size, num_window, seq_length = bc.get_shape_list2(stacked_input_ids)
        window_vectors = tf.reshape(window_vectors, [batch_size, num_window, -1])
        window_vectors = window_vectors * tf.cast(tf.reshape(use_context, [-1, 1, 1]), tf.float32)
        window_vectors = tf.tile(window_vectors, [1, num_window, 1]) # [batch_size, num_window * num_window, hidden_size]
        window_vectors = tf.reshape(window_vectors, [batch_size * num_window, num_window, -1])
        print("window_vectors", window_vectors.shape)
        input_vectors = tf.concat([lower_module_last_layer, window_vectors], axis=1)
        # input_vectors : [batch_size * num_window, window_length + num_window, hidden_size]
        print("input_vectors", input_vectors.shape)
        added_tokens = num_window
        attention_mask = tf.pad(self.lower_module.attention_mask,
                                [[0, 0], [0, added_tokens], [0, added_tokens]], 'CONSTANT', constant_values=1)
        with tf.compat.v1.variable_scope("mid"):
            self.mid_layers.call(input_vectors, attention_mask)
            middle_output = self.mid_layers.get_last_layer_output()
        print("middle_output ", middle_output.shape)

        middle_output_head = middle_output[:, :self.window_size, :]

        # Now re-distribute the context
        middle_output_tail = middle_output[:, self.window_size:, :]  # [batch_size * num_window, num_window, hidden_size]
        middle_output_tail = tf.reshape(middle_output_tail , [batch_size, num_window, num_window, -1])
        # tail[batch_idx, idx1, idx2, :] : response value to window[idx2] from window[idx1]
        # -> This is what window[idx2] may want to know about window[idx1]
        middle_output_tail = tf.transpose(middle_output_tail, [0,2,1,3])
        middle_output_tail = tf.reshape(middle_output_tail, [batch_size * num_window, num_window, -1])

        input_to_upper = tf.concat([middle_output_head,middle_output_tail], axis=1)
        with tf.compat.v1.variable_scope("upper"):
            self.upper_module.call(input_to_upper, attention_mask)
        self.embedding_table = self.lower_module.embedding_layer.embedding_table
        raw_sequence_output = self.upper_module.all_layer_outputs[-1]
        self.sequence_output = raw_sequence_output[:, :self.window_size, :]
        self.all_encoder_layers = self.lower_module.all_layer_outputs + self.upper_module.all_layer_outputs

        self.all_encoder_layers = []
        self.embedding_output = self.lower_module.embedding_output
        return self.sequence_output


class MidModuleExpanding:
    def __init__(self, old_config: JsonConfig):
        self.n_layers = 3
        self.all_layer_outputs = []
        self.last_key_layer = None
        self.old_config = old_config

        self.inner_config = self.build_config(old_config, old_config.mid_expanding_factor)
        self.embedding = None
        self.layer_list = []
        self.initializer = base.create_initializer(self.inner_config.initializer_range)
        self.token_type_table = tf.compat.v1.get_variable(
                name="token_type_embeddings",
                shape=[self.inner_config.type_vocab_size, self.inner_config.hidden_size],
                initializer=self.initializer)
        self.full_position_embeddings = tf.compat.v1.get_variable(
            name="position_embeddings",
            shape=[self.inner_config.max_position_embeddings, self.inner_config.hidden_size],
            initializer=self.initializer)
        with tf.compat.v1.variable_scope("mid"):
            for layer_idx in range(self.n_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    layer = ForwardLayer(self.inner_config, self.initializer)
                    self.layer_list.append(layer)

    @staticmethod
    def build_config(config:JsonConfig, expanding_factor):
        new_config = JsonConfig.from_dict(config.to_dict())
        hidden_size = config.hidden_size * expanding_factor
        new_config.set_attrib("hidden_size", hidden_size)
        intermediate_size = config.intermediate_size * expanding_factor
        new_config.set_attrib("intermediate_size ", intermediate_size )
        return new_config

    def expand_vectors(self, input_vectors):
        # input_vectors : [seq_length, hidden_size]
        seq_length, hidden_dim = bc.get_shape_list2(input_vectors)
        pad_size = self.inner_config.hidden_size - hidden_dim
        pad = tf.zeros([seq_length, pad_size])
        return tf.concat([input_vectors, pad], axis=1)

    def call(self, input_vectors, use_context):
        # input_vectors : [num_window, hidden_size]
        input_vectors = self.expand_vectors(input_vectors)
        seq_length, hidden_dim = bc.get_shape_list2(input_vectors)
        # Add position embedding
        input_vectors = tf.expand_dims(input_vectors, 0)
        input_vectors = bc.embedding_postprocessor2(
            input_tensor=input_vectors,
            token_type_table=self.token_type_table,
            full_position_embeddings=self.full_position_embeddings,
            use_token_type=False,
            token_type_ids=None,
            token_type_vocab_size=1,
            use_position_embeddings=True,
            max_position_embeddings=self.inner_config.max_num_window,
            dropout_prob=self.inner_config.hidden_dropout_prob)

        input_shape = [1, seq_length]

        attention_mask = tf.ones([1, seq_length, seq_length], tf.int32) * use_context
        with tf.compat.v1.variable_scope("mid"):
            prev_output = bc.reshape_to_matrix(input_vectors)
            for layer_idx in range(self.n_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = self.layer_list[layer_idx].apply(prev_output, 1, seq_length,
                                                             attention_mask)
                    final_output = bc.reshape_from_matrix2(prev_output, input_shape)
                    self.all_layer_outputs.append(final_output)

        prev_output = tf.reshape(prev_output,
                                 [seq_length, self.inner_config.mid_expanding_factor, self.old_config.hidden_size])
        return prev_output

    def pad_attention_mask(self, attention_mask):
        added_tokens = self.inner_config.mid_expanding_factor - 1
        return tf.pad(attention_mask, [[0,0], [0, added_tokens], [0, added_tokens]], 'CONSTANT', constant_values=1)


class MidModule:
    def __init__(self, config):
        self.n_layers = 3 #
        self.all_layer_outputs = []
        self.last_key_layer = None
        self.config = config
        self.embedding = None
        self.layer_list = []
        self.initializer = base.create_initializer(config.initializer_range)
        self.token_type_table = tf.compat.v1.get_variable(
                name="token_type_embeddings",
                shape=[config.type_vocab_size, config.hidden_size],
                initializer=self.initializer)
        self.full_position_embeddings = tf.compat.v1.get_variable(
            name="position_embeddings",
            shape=[config.max_position_embeddings, config.hidden_size],
            initializer=self.initializer)
        with tf.compat.v1.variable_scope("mid"):
            for layer_idx in range(self.n_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    layer = ForwardLayer(self.config, self.initializer)
                    self.layer_list.append(layer)

    def call(self, input_vectors, use_context):
        # input_vectors : [num_window, hidden_size]
        seq_length, hidden_dim = bc.get_shape_list2(input_vectors)
        # Add position embedding
        input_vectors = tf.expand_dims(input_vectors, 0)
        input_vectors = bc.embedding_postprocessor2(
            input_tensor=input_vectors,
            token_type_table=self.token_type_table,
            full_position_embeddings=self.full_position_embeddings,
            use_token_type=False,
            token_type_ids=None,
            token_type_vocab_size=1,
            use_position_embeddings=True,
            max_position_embeddings=self.config.max_num_window,
            dropout_prob=self.config.hidden_dropout_prob)

        input_shape = [1, seq_length]

        attention_mask = tf.ones([1, seq_length, seq_length], tf.int32) * use_context
        with tf.compat.v1.variable_scope("mid"):
            prev_output = bc.reshape_to_matrix(input_vectors)
            for layer_idx in range(self.n_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = self.layer_list[layer_idx].apply(prev_output, 1, seq_length,
                                                             attention_mask)
                    final_output = bc.reshape_from_matrix2(prev_output, input_shape)
                    self.all_layer_outputs.append(final_output)

        return prev_output

    def concat_all_layers(self):
        return tf.stack(self.all_layer_outputs, axis=1)


class MidModuleGamma:
    def __init__(self, config):
        self.n_layers = 3 #
        self.all_layer_outputs = []
        self.last_key_layer = None
        self.config = config
        self.embedding = None
        self.layer_list = []
        self.initializer = base.create_initializer(config.initializer_range)
        self.token_type_table = tf.compat.v1.get_variable(
                name="token_type_embeddings",
                shape=[config.type_vocab_size, config.hidden_size],
                initializer=self.initializer)
        self.full_position_embeddings = tf.compat.v1.get_variable(
            name="position_embeddings",
            shape=[config.max_position_embeddings, config.hidden_size],
            initializer=self.initializer)
        with tf.compat.v1.variable_scope("mid"):
            for layer_idx in range(self.n_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    layer = ForwardLayer(self.config, self.initializer)
                    self.layer_list.append(layer)

    def call(self, input_vectors, use_context):
        # input_vectors : [num_window, hidden_size]
        batch_size, seq_length, hidden_dim = bc.get_shape_list2(input_vectors)
        # Add position embedding
        input_vectors = bc.embedding_postprocessor2(
            input_tensor=input_vectors,
            token_type_table=self.token_type_table,
            full_position_embeddings=self.full_position_embeddings,
            use_token_type=False,
            token_type_ids=None,
            token_type_vocab_size=1,
            use_position_embeddings=True,
            max_position_embeddings=self.config.max_num_window,
            dropout_prob=self.config.hidden_dropout_prob)

        input_shape = [batch_size, seq_length]

        attention_mask = tf.ones([batch_size, seq_length, seq_length], tf.int32) * tf.expand_dims(use_context, 2)
        with tf.compat.v1.variable_scope("mid"):
            prev_output = bc.reshape_to_matrix(input_vectors)
            for layer_idx in range(self.n_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = self.layer_list[layer_idx].apply(prev_output, batch_size, seq_length,
                                                             attention_mask)
                    final_output = bc.reshape_from_matrix2(prev_output, input_shape)
                    self.all_layer_outputs.append(final_output)

        return prev_output

    def concat_all_layers(self):
        return tf.stack(self.all_layer_outputs, axis=1)



def check_SeroAlpha():
    dummy_input_ids = tf.ones([128 * 128], tf.int64)
    dummy_input_mask = tf.ones([128 * 128], tf.int64)
    from path import data_path
    config = JsonConfig.from_json_file(os.path.join(data_path, "config", "sero.json"))
    model = SeroAlpha(config, True, dummy_input_ids, dummy_input_mask, dummy_input_mask)

    print(model.get_sequence_output())


if __name__ == "__main__":
    check_SeroAlpha()