from functools import partial

import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_ranking.python.data import build_ranking_dataset_with_parsing_fn

from misc.ranking_input_fn import _LABEL_FEATURE, group_size
from models.transformer import optimization_v2
from tlm.training.input_fn_common import _decode_record


def parsing_fn_w_seq_len(max_seq_length, record):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }
    return _decode_record(record, name_to_features),


def make_input_fn(file_pattern,
                  FLAGS,
                  randomize_input=True,
                  num_epochs=None):
  """Returns `Estimator` `input_fn` for TRAIN and EVAL.

  Args:
    file_pattern: (string) file pattern for the TFRecord input data.
    batch_size: (int) number of input examples to process per batch.
    randomize_input: (bool) if true, randomize input example order. It should
      almost always be true except for unittest/debug purposes.
    num_epochs: (int) Number of times the input dataset must be repeated. None
      to repeat the data indefinitely.

  Returns:
    An `input_fn` for `Estimator`.
  """
  batch_size = FLAGS.train_batch_size
  parsing_fn = partial(parsing_fn_w_seq_len, FLAGS.max_seq_length)
  def _input_fn():
    """Defines the input_fn."""
    dataset = build_ranking_dataset_with_parsing_fn(
        file_pattern, parsing_fn=parsing_fn, batch_size=batch_size)

    features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
    label = tf.cast(label, tf.float32)

    # Add document_mask to features, which is True for valid documents and False
    # for invalid documents.
    return features, label

  return _input_fn


def make_score_fn(bert_config, model_class, train_config, is_training, special_flags=[]):
  """Returns a scoring function to build `EstimatorSpec`."""

  def _score_fn(context_features, group_features, mode, params, config):
    """Defines the network to score a group of documents."""
    del [params, config]

    features = group_features
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    model = model_class(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=train_config.use_one_hot_embeddings,
    )
    pooled = model.get_pooled_output()
    logits = tf.compat.v1.layers.dense(pooled, units=group_size)
    return logits

  return _score_fn


def make_transform_fn():
  """Returns a transform_fn that converts features to dense Tensors."""

  def _transform_fn(features, mode):
    """Defines transform_fn."""
    return features, []

  return _transform_fn


def eval_metric_fns():
  """Returns a dict from name to metric functions."""
  metric_fns = {}
  metric_fns.update({
      "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
          tfr.metrics.RankingMetricKey.ARP,
          tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
      ]
  })
  metric_fns.update({
      "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.NDCG, topn=topn)
      for topn in [1, 3, 5, 10]
  })
  for topn in [1, 3, 5, 10]:
    metric_fns["metric/weighted_ndcg@%d" % topn] = (
        tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn))
  return metric_fns


def _train_op_fn(train_config, loss):
    train_op = optimization_v2.create_optimizer_from_config(loss, train_config)
    return train_op


def get_model_fn(score_fn, train_config):


    loss_type = "approx_ndcg_loss"
    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(loss_type),
        eval_metric_fns=eval_metric_fns(),
        train_op_fn=lambda x:_train_op_fn(train_config, x))

    model_fn = tfr.model.make_groupwise_ranking_fn(
          group_score_fn=score_fn,
          group_size=group_size,
          transform_fn=make_transform_fn(),
          ranking_head=ranking_head)
    return model_fn