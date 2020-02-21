from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(serialized=record, features=name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, dtype=tf.int32)
    example[name] = t

  return example


def input_fn(mode, input_context=None):
  # mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
  #                  datasets['test'])
  # mnist_dataset = tf.data.Dataset.from_tensor_slices([numpy.zeros([28,28,1]), 0] * 100)
  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([512], tf.int64),
      "input_mask": tf.io.FixedLenFeature([512], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([512], tf.int64),
      "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
  }
  input_file_path = "/home/youngwookim/code/Chair/data/ukp_1K/abortion"
  d = tf.data.TFRecordDataset(input_file_path)
  d = d.apply(
      tf.data.experimental.map_and_batch(
          lambda record: _decode_record(record, name_to_features),
          batch_size=64,
          num_parallel_batches=8,
          drop_remainder=True))

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  if input_context:
    d = d.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
  return d


LEARNING_RATE = 1e-4
def model_fn(features, labels, mode):

  def model(featrues, training):
    t = featrues["input_ids"][:, :2]
    t = tf.cast(t, tf.float32)
    return tf.keras.layers.Dense(10)(t)

  logits = model(features, training=False)
  labels = tf.zeros([64, 1], tf.int32)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'logits': logits}
    return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

  optimizer = tf.compat.v1.train.GradientDescentOptimizer(
      learning_rate=LEARNING_RATE)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
  loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=optimizer.minimize(
          loss, tf.compat.v1.train.get_or_create_global_step()))


strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy,
                                save_checkpoints_secs=None,
                                save_checkpoints_steps=1000,
                                )
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

model_dir = os.path.join('multiworker')
classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=model_dir, config=config)

print("Starting training")
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn, max_steps=10000),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)

