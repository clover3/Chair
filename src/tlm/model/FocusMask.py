from tlm.model.base import *


class FocusMask(BertModelInterface):
  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               scope=None,
               features=None,
               ):
      super(FocusMask, self).__init__()
      config = copy.deepcopy(config)
      self.config = config
      if not is_training:
          config.hidden_dropout_prob = 0.0
          config.attention_probs_dropout_prob = 0.0

      input_shape = get_shape_list(input_ids, expected_rank=2)
      batch_size = input_shape[0]
      seq_length = input_shape[1]
      focus_mask = features['focus_mask']

      if input_mask is None:
          input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

      if token_type_ids is None:
          token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

      with tf.compat.v1.variable_scope(scope, default_name="bert"):
          with tf.compat.v1.variable_scope("embeddings"):
              # Perform embedding lookup on the word ids.
              (self.embedding_output, self.embedding_table) = embedding_lookup(
                  input_ids=input_ids,
                  vocab_size=config.vocab_size,
                  embedding_size=config.hidden_size,
                  initializer_range=config.initializer_range,
                  word_embedding_name="word_embeddings",
                  use_one_hot_embeddings=use_one_hot_embeddings)

              # Add positional embeddings and token type embeddings, then layer
              # normalize and perform dropout.
              self.embedding_output = embedding_postprocessor(
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
              self.embedding_output = embedding_postprocessor(
                  input_tensor=self.embedding_output,
                  use_token_type=True,
                  token_type_ids=focus_mask,
                  token_type_vocab_size=4,
                  token_type_embedding_name="focus_embedding",
                  use_position_embeddings=False,
                  position_embedding_name=None,
                  initializer_range=config.initializer_range,
                  max_position_embeddings=config.max_position_embeddings,
                  dropout_prob=config.hidden_dropout_prob)

          with tf.compat.v1.variable_scope("encoder"):
              attention_mask = create_attention_mask_from_input_mask(
                  input_ids, input_mask)

              self.all_encoder_layers = transformer_model(
                  input_tensor=self.embedding_output,
                  attention_mask=attention_mask,
                  input_mask=input_mask,
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

