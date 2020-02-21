from abc import ABC, abstractmethod

import tensorflow as tf

import models.transformer.bert_common_v2 as bc
from tlm.model import base
from tlm.model.base import mimic_pooling
from tlm.model.units import ForwardLayer, Embedding
from tlm.model_cnfig import JsonConfig


def get_per_layer_config(all_config:JsonConfig):
    keys = ["attention_probs_dropout_prob", "hidden_act", "hidden_dropout_prob", "hidden_size",
              "initializer_range", "intermediate_size", "num_attention_heads"]

    d = {}
    for key in keys:
        d[key] = all_config.__dict__[key]

    return JsonConfig.from_dict(d)


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
        self.layer_idx_base = 0

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
            with tf.compat.v1.variable_scope("layer_%d" % (layer_idx + self.layer_idx_base)):
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
    stacked_input_mask = tf.stack(window_input_mask_list, 1)  # [batch_size, num_window, src_window_length]
    stacked_segment_ids = tf.stack(window_segment_ids_list, 1)  # [batch_size, num_window, src_window_length]

    batch_size, num_window, _ = bc.get_shape_list2(stacked_input_ids)
    edge_shape = [batch_size, num_window, 1]
    cls_arr = tf.ones(edge_shape, tf.int32) * 23
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
        self.upper_module = UpperTransformer(config, config.upper_layers)
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
        context_vectors = self.mid_layers.call(window_vectors, use_context) # [batch_size, num_window, hidden_size ]
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


def apply_cotext_mask(info_vectors, use_context):
    dims = len(info_vectors.shape)
    if dims == 3:
        broadcast_shape = [-1, 1, 1]
    elif dims == 4:
        broadcast_shape = [-1, 1, 1, 1]
    info_vectors = info_vectors * tf.cast(tf.reshape(use_context, broadcast_shape), tf.float32)
    return info_vectors

def exchange_return_context(batch_size, middle_output, window_size, num_window, use_context):
    middle_output_head = middle_output[:, :window_size, :]
    # Now re-distribute the context
    middle_output_tail = middle_output[:, window_size:,
                         :]  # [batch_size * num_window, num_window, hidden_size]
    middle_output_tail = tf.reshape(middle_output_tail, [batch_size, num_window, num_window, -1])
    # tail[batch_idx, idx1, idx2, :] : response value to window[idx2] from window[idx1]
    # -> This is what window[idx2] may want to know about window[idx1]
    middle_output_tail = tf.transpose(middle_output_tail, [0, 2, 1, 3])
    middle_output_tail = apply_cotext_mask(middle_output_tail , use_context)
    middle_output_tail = tf.reshape(middle_output_tail, [batch_size * num_window, num_window, -1])
    input_to_upper = tf.concat([middle_output_head, middle_output_tail], axis=1)
    return input_to_upper

def exchange_contexts(batch_size, lower_module_last_layer, num_window, use_context):
    window_vectors = lower_module_last_layer[:, -1, :]
    window_vectors = tf.reshape(window_vectors, [batch_size, num_window, -1])
    window_vectors = apply_cotext_mask(window_vectors, use_context)
    window_vectors = tf.tile(window_vectors,
                             [1, num_window, 1])  # [batch_size, num_window * num_window, hidden_size]
    window_vectors = tf.reshape(window_vectors, [batch_size * num_window, num_window, -1])
    input_vectors = tf.concat([lower_module_last_layer, window_vectors], axis=1)
    return input_vectors


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
        batch_size, num_window, seq_length = bc.get_shape_list2(stacked_input_ids)

        self.lower_module.call(r3to2(stacked_input_ids),
                               r3to2(stacked_input_mask),
                               r3to2(stacked_segment_ids),
                               )

        lower_module_last_layer = self.lower_module.all_layer_outputs[
            -1]  # [ batch_size * num_window, seq_length, hidden_size)
        input_vectors = exchange_contexts(batch_size, lower_module_last_layer, num_window, use_context)
        # input_vectors : [batch_size * num_window, window_length + num_window, hidden_size]
        added_tokens = num_window
        attention_mask = tf.pad(self.lower_module.attention_mask,
                                [[0, 0], [0, added_tokens], [0, added_tokens]], 'CONSTANT', constant_values=1)
        with tf.compat.v1.variable_scope("mid"):
            self.mid_layers.call(input_vectors, attention_mask)
            middle_output = self.mid_layers.get_last_layer_output()

        input_to_upper = exchange_return_context(batch_size, middle_output,
                                                      self.window_size, num_window, use_context)
        with tf.compat.v1.variable_scope("upper"):
            self.upper_module.call(input_to_upper, attention_mask)
        self.embedding_table = self.lower_module.embedding_layer.embedding_table
        raw_sequence_output = self.upper_module.all_layer_outputs[-1]
        self.sequence_output = raw_sequence_output[:, :self.window_size, :]
        self.all_encoder_layers = self.lower_module.all_layer_outputs + self.upper_module.all_layer_outputs

        self.all_encoder_layers = []
        self.embedding_output = self.lower_module.embedding_output
        return self.sequence_output


class SeroEpsilon(base.BertModelInterface):
    def __init__(self,
                 config,  # This is different from BERT config,
                 is_training,
                 use_one_hot_embeddings=True,
                 ):
        super(SeroEpsilon, self).__init__()


        if not is_training:
            config.set_attrib("hidden_dropout_prob", 0.0)
            config.set_attrib("attention_probs_dropout_prob", 0.0)
        self.is_training = is_training
        self.total_sequence_length = config.total_sequence_length
        self.window_size = config.window_size
        self.initializer_range = config.initializer_range
        self.lower_module = LowerTransformer(config, config.lower_layers, use_one_hot_embeddings)
        self.hidden_size = config.hidden_size
        per_layer_config = get_per_layer_config(config)
        self.upper_module_list = []
        for i in range(0, config.upper_layers, 2):
            ut = UpperTransformer(per_layer_config, 2)
            ut.layer_idx_base = i
            self.upper_module_list.append(ut)
        self.pooling = config.pooling
        self.upper_module_inputs = []


    def network_stacked(self, stacked_input_ids, stacked_input_mask,
                        stacked_segment_ids, use_context):
        batch_size, num_window, seq_length = bc.get_shape_list2(stacked_input_ids)
        self.batch_size = batch_size
        self.num_window = num_window

        self.lower_module.call(r3to2(stacked_input_ids),
                               r3to2(stacked_input_mask),
                               r3to2(stacked_segment_ids),
                               )

        lower_module_last_layer = self.lower_module.all_layer_outputs[-1]  # [ batch_size * num_window, seq_length, hidden_size)
        input_to_upper = exchange_contexts(batch_size, lower_module_last_layer, num_window, use_context)
        # input_vectors : [batch_size * num_window, window_length + num_window, hidden_size]
        added_tokens = num_window
        attention_mask = tf.pad(self.lower_module.attention_mask,
                                [[0, 0], [0, added_tokens], [0, added_tokens]], 'CONSTANT', constant_values=1)
        with tf.compat.v1.variable_scope("upper"):
            for upper_module in self.upper_module_list:
                self.upper_module_inputs.append(input_to_upper)
                upper_module.call(input_to_upper, attention_mask)
                middle_output = upper_module.get_last_layer_output()
                input_to_upper = exchange_return_context(batch_size, middle_output, self.window_size,
                                                         num_window, use_context)

        self.embedding_table = self.lower_module.embedding_layer.embedding_table
        raw_sequence_output = self.upper_module_list[-1].all_layer_outputs[-1]
        self.sequence_output = raw_sequence_output[:, :self.window_size, :]
        self.all_encoder_layers = self.lower_module.all_layer_outputs
        for upper_module in self.upper_module_list:
            self.all_encoder_layers.extend(upper_module.all_layer_outputs)

        self.all_encoder_layers = []
        self.embedding_output = self.lower_module.embedding_output
        if self.pooling == "head":
            self.pooled_output = self.head_pooling()
        elif self.pooling == "all":
            self.pooled_output = self.all_pooling()
        elif self.pooling == "none":
            pass

        return self.sequence_output

    def head_pooling(self):
        seq_4d = tf.reshape(self.sequence_output, [self.batch_size, self.num_window, self.window_size, self.hidden_size])
        first_sequence_output = seq_4d[:, 0]
        pooled_output = mimic_pooling(first_sequence_output, self.hidden_size, self.initializer_range)
        return pooled_output

    def all_pooling(self):
        seq_4d = tf.reshape(self.sequence_output, [self.batch_size, self.num_window, self.window_size, self.hidden_size])
        seq_4d = tf.transpose(seq_4d, [0, 2, 1, 3])
        pooled_output = mimic_pooling(seq_4d, self.hidden_size, self.initializer_range)
        pooled_output = tf.reduce_max(pooled_output, axis=2)
        return pooled_output


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


class SeroZeta(base.BertModelInterface):
    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(SeroZeta, self).__init__()

        if not is_training:
            config.set_attrib("hidden_dropout_prob", 0.0)
            config.set_attrib("attention_probs_dropout_prob", 0.0)

        self.total_sequence_length = config.total_sequence_length
        self.window_size = config.window_size

        self.lower_module = LowerTransformer(config, config.num_hidden_layers, use_one_hot_embeddings)

        if config.pooling == "lstm":
            self.combine_model = LSTMCombiner(config.hidden_size)
        elif config.pooling == "bilstm":
            self.combine_model = BiLSTMCombiner(config.hidden_size)
        elif config.pooling.startswith("lstm"):
            self.combine_model = SubLSTMCombiner(config.pooling, config.hidden_size)
        elif config.pooling == "first":
            self.combine_model = FirstTaker(config.hidden_size)
        else:
            print("config.pooling is not specified default to LSTMCombiner")
            self.combine_model = LSTMCombiner(config.hidden_size)


    def network_stacked(self, stacked_input_ids, stacked_input_mask,
                 stacked_segment_ids, use_context):
        batch_size, num_window, seq_length = bc.get_shape_list2(stacked_input_ids)
        self.lower_module.call(r3to2(stacked_input_ids),
                               r3to2(stacked_input_mask),
                               r3to2(stacked_segment_ids),
                               )

        lower_module_last_layer = self.lower_module.all_layer_outputs[-1]
        #[ batch_size * num_window, seq_length, hidden_size)
        lower_module_last_layer = tf.reshape(lower_module_last_layer, [batch_size, num_window, seq_length, -1])

        self.pooled_output = self.combine_model.call(lower_module_last_layer)
        print(self.pooled_output)

        self.embedding_table = self.lower_module.embedding_layer.embedding_table
        self.sequence_output = lower_module_last_layer
        self.all_encoder_layers = self.lower_module.all_layer_outputs
        self.embedding_output = self.lower_module.embedding_output
        return self.sequence_output


class CombinerInterface(ABC):
    @abstractmethod
    def call(self, input_tensor):
        # input_tensor : [batch_size, num_window, seq_length, hidden_size]
        pass


class LSTMCombiner:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
    def call(self, input_tensor):
        lstm = tf.compat.v1.keras.layers.LSTM(self.hidden_size, return_sequences=True, return_state=True)
        first_tokens = input_tensor[:, :, 0, :]
        print(input_tensor.shape)
        whole_seq_output, final_memory_state, final_carry_state = lstm(first_tokens)

        last_output = whole_seq_output[:, -1]
        return last_output


class SubLSTMCombiner:
    def __init__(self, pool_method, hidden_size):
        pool_loc = int(pool_method[4:])
        self.hidden_size = hidden_size
        self.pool_loc = pool_loc
        print("Pool loc : ", pool_loc)

    def call(self, input_tensor):
        lstm = tf.compat.v1.keras.layers.LSTM(self.hidden_size, return_sequences=True, return_state=True)
        first_tokens = input_tensor[:, :, 0, :]
        whole_seq_output, final_memory_state, final_carry_state = lstm(first_tokens)

        last_output = whole_seq_output[:, self.pool_loc]
        return last_output



class BiLSTMCombiner:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
    def call(self, input_tensor):
        lstm_fw = tf.compat.v1.keras.layers.LSTM(self.hidden_size, return_sequences=True, return_state=True)
        lstm_bw = tf.compat.v1.keras.layers.LSTM(self.hidden_size, return_sequences=True, return_state=True)

        first_tokens = input_tensor[:, :, 0, :]
        whole_seq_output_fw, final_memory_state, final_carry_state = lstm_fw(first_tokens)
        whole_seq_output_bw, _, _ = lstm_bw(first_tokens[:, ::-1, :])

        h1 = tf.concat([whole_seq_output_fw[:, -1], whole_seq_output_bw[:, -1]], axis=1)
        pooled_output = tf.keras.layers.Dense(self.hidden_size, activation=tf.nn.relu)(h1)
        return pooled_output

class FirstTaker:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def call(self, input_tensor):
        first_tokens = input_tensor[:, :, 0, :]
        tf.keras.layers.Dense(self.hidden_size, activation=tf.nn.relu)(first_tokens)
        return first_tokens[:,0]

def try_lstm_combiner():
    hidden_size = 128
    combiner = LSTMCombiner(hidden_size)

    input_tensor = tf.ones([10, 8, 20, hidden_size])
    pooled = combiner.call(input_tensor)
    print(pooled.shape)

    print(pooled)


if __name__ == "__main__":
    try_lstm_combiner()