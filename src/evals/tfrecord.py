from tf_util.enum_features import load_record


def load_tfrecord(record_path):
    for feature in load_record(record_path):
        input_ids = feature["input_ids"].int64_list.value
        label_ids = feature["label_ids"].int64_list.value[0]
        yield input_ids, label_ids