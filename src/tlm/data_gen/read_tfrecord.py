from my_tf import tf


def extract_stream(fn):
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature

        def take(v):
            return v.int64_list.value[:200]

        input_ids = take(feature["input_ids"])
        segment_ids = take(feature["segment_ids"])
        input_mask = take(feature["input_mask"])

        yield input_ids, input_mask, segment_ids


# extract_stream("C:\work\Code\Chair\data\\tf\\0")