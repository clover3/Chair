import tensorflow as tf
from official.nlp.bert.input_pipeline import single_file_dataset


def enum_feature_names():
    for seg_name in ["premise", "hypothesis"]:
        for sub_name in ["input_ids", "attention_mask"]:
            feature_name = f"{seg_name}_{sub_name}"
            yield feature_name



def create_dataset_sbert_text(file_path,
                              batch_size,
                              is_training=True,
                              input_pipeline_context=None,
                              label_type=tf.int64,
                              include_sample_weights=False,
                              num_samples=None):
    """Creates input dataset from (tf)records files for train/eval."""
    text_spec = tf.io.VarLenFeature(tf.string)
    name_to_features = {
        'label': tf.io.FixedLenFeature([], label_type),
    }
    for feature_name in enum_feature_names():
        name_to_features[feature_name] = text_spec

    dataset = single_file_dataset(file_path, name_to_features,
                                  num_samples=num_samples)
    def parse_tensors(record):
        for feature_name in enum_feature_names():
            v = record[feature_name]
            v = tf.sparse.to_dense(v)[0]
            v = tf.io.parse_tensor(v, tf.int32)
            v = v[:16, :16]
            pad_len = 16 - tf.shape(v)[0]
            pad_len2 = 16 - tf.shape(v)[1]
            v = tf.pad(v, [[0, pad_len], [0, pad_len2]], 'CONSTANT')
            record[feature_name] = v
            # record[feature_name] = parse(v)
        return record

    dataset = dataset.map(parse_tensors)
    dataset = dataset.take(1000)
    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                input_pipeline_context.input_pipeline_id)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def get_dataset_fn(input_file_pattern,
                   global_batch_size,
                   is_training,
                   label_type=tf.int64,
                   include_sample_weights=False,
                   num_samples=None):
    """Gets a closure to create a dataset."""

    def _dataset_fn(ctx=None):
        """Returns tf.data.Dataset for distributed BERT pretraining."""
        batch_size = ctx.get_per_replica_batch_size(
            global_batch_size) if ctx else global_batch_size
        dataset = create_dataset_sbert_text(
            tf.io.gfile.glob(input_file_pattern),
            batch_size,
            is_training=is_training,
            input_pipeline_context=ctx,
            label_type=label_type,
            include_sample_weights=include_sample_weights,
            num_samples=num_samples)
        return dataset

    return _dataset_fn
