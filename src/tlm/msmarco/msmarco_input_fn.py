import tensorflow as tf

from data_generator.special_tokens import CLS_ID, SEP_ID


def input_fn_builder(input_files, seq_length, is_training,
                     max_eval_examples=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
    output_buffer_size = batch_size * 1000

    def extract_fn(data_record):
      features = {
          "query_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
          "doc_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
          "label": tf.io.FixedLenFeature([], tf.int64),
      }
      sample = tf.compat.v1.parse_single_example(data_record, features)
      query_ids = tf.cast(sample["query_ids"], tf.int32)
      doc_ids = tf.cast(sample["doc_ids"], tf.int32)
      label_ids = tf.cast(sample["label"], tf.int32)
      input_ids = tf.concat([[CLS_ID], query_ids, [SEP_ID], doc_ids, [SEP_ID]], 0)

      query_segment_id = tf.zeros_like(query_ids)
      doc_segment_id = tf.ones_like(doc_ids)
      segment_ids = tf.concat(([0], query_segment_id, [0], doc_segment_id, [1]), 0)
      input_mask = tf.ones_like(input_ids)

      features = {
          "input_ids": input_ids,
          "segment_ids": segment_ids,
          "input_mask": input_mask,
          "label_ids": label_ids,
      }
      return features

    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(
        extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=1000)
    else:
      if max_eval_examples:
        # Use at most this number of examples (debugging only).
        dataset = dataset.take(max_eval_examples)
        # pass

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes={
            "input_ids": [seq_length],
            "segment_ids": [seq_length],
            "input_mask": [seq_length],
            "label_ids": [],
        },
        padding_values={
            "input_ids": 0,
            "segment_ids": 0,
            "input_mask": 0,
            "label_ids": 0,
        },
        drop_remainder=True)

    return dataset
  return input_fn



