import copy

from models.transformer.bert_common_v2 import *
from models.transformer.bert_common_v2 import create_initializer
from tlm.model.base import BertModelInterface, transformer_model, BertModel
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2


class DoubleLengthInputModel(BertModelInterface):
    def __init__(self,
               config,
               is_training,
               input_ids1,
               input_mask1,
               token_type_ids1,
               input_ids2,
               input_mask2,
               token_type_ids2,
               use_one_hot_embeddings=True,
               features=None,
               scope=None):
        super(DoubleLengthInputModel, self).__init__()
        input_shape = get_shape_list(input_ids1, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # feed input separtely to the network
        config = copy.deepcopy(config)
        self.config = config
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        batch_concat_input_ids = tf.concat([input_ids1, input_ids2], 0) # [ batch_size * 2,  seq_length]
        batch_concat_concat_token_ids = tf.concat([token_type_ids1, token_type_ids2], 0)
        input_mask_seq_concat = tf.concat([input_mask1, input_mask2], 1)

        with tf.compat.v1.variable_scope(scope, default_name="bert"):
            with tf.compat.v1.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (embedding_output_batch_concat, self.embedding_table) = embedding_lookup(
                    input_ids=batch_concat_input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                embedding_output_batch_concat = embedding_postprocessor(
                    input_tensor=embedding_output_batch_concat,
                    use_token_type=True,
                    token_type_ids=batch_concat_concat_token_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            embedding_output_stacked = tf.reshape(embedding_output_batch_concat,
                                                  [2, batch_size, seq_length, -1])
            embedding_output_stacked = tf.transpose(embedding_output_stacked, [1, 0, 2, 3])
            embedding_output_seq_concat = tf.reshape(embedding_output_stacked, [batch_size, seq_length * 2, -1])
            self.embedding_output = embedding_output_seq_concat

            with tf.compat.v1.variable_scope("encoder"):
                attention_mask = create_attention_mask_from_input_mask(
                    input_mask_seq_concat, input_mask_seq_concat)

                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    input_mask=input_mask_seq_concat,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    is_training=is_training,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            with tf.compat.v1.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.keras.layers.Dense(config.hidden_size,
                                                           activation=tf.keras.activations.tanh,
                                                           kernel_initializer=create_initializer(config.initializer_range))(
                    first_token_tensor)


class DualBertTwoInputWithDoubleInputLength(BertModelInterface):
    def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               features=None,
               scope=None):
        super(DualBertTwoInputWithDoubleInputLength, self).__init__()

        input_ids1 = features["input_ids1"]
        input_mask1 = features["input_mask1"]
        segment_ids1 = features["segment_ids1"]
        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model_1 = BertModel(
                    config=config,
                    is_training=is_training,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids,
                    use_one_hot_embeddings=use_one_hot_embeddings,
            )

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model_2 = DoubleLengthInputModel(
                    config,
                    is_training,
                    input_ids1,
                    input_mask1,
                    segment_ids1,
                    input_ids2,
                    input_mask2,
                    segment_ids2,
                    use_one_hot_embeddings=use_one_hot_embeddings,
            )

        model_1_first_token = model_1.get_sequence_output()[:, 0, :]
        model_2_first_token = model_2.get_sequence_output()[:, 0, :]

        rep = tf.concat([model_1_first_token, model_2_first_token], axis=1)

        self.sequence_output = model_1.get_sequence_output()
        dense_layer = tf.keras.layers.Dense(config.hidden_size,
                                              activation=tf.keras.activations.tanh,
                                              kernel_initializer=create_initializer(config.initializer_range))
        pooled_output = dense_layer(rep)
        self.pooled_output = pooled_output


class DualBertTwoInputWithDoubleInputLengthTakeSecond(BertModelInterface):
    def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               features=None,
               scope=None):
        super(DualBertTwoInputWithDoubleInputLengthTakeSecond, self).__init__()

        input_ids1 = features["input_ids1"]
        input_mask1 = features["input_mask1"]
        segment_ids1 = features["segment_ids1"]
        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        model = DoubleLengthInputModel(
                config,
                is_training,
                input_ids1,
                input_mask1,
                segment_ids1,
                input_ids2,
                input_mask2,
                segment_ids2,
                use_one_hot_embeddings=use_one_hot_embeddings,
        )

        self.sequence_output = model.get_sequence_output()
        self.pooled_output = model.get_pooled_output()
