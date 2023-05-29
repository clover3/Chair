import sys

import tensorflow as tf



def main():
    max_seq_length = 300
    def decode_record(record):
        name_to_features = {
        }
        def fixed_len_feature():
            return tf.io.FixedLenFeature([max_seq_length], tf.int64)
        name_to_features[f'input_ids'] = fixed_len_feature()
        name_to_features[f'segment_ids'] = fixed_len_feature()
        name_to_features[f'label_ids'] = tf.io.FixedLenFeature([1], tf.int64)
        tli_label_len = max_seq_length * 3
        name_to_features[f'tli_label'] = tf.io.FixedLenFeature([tli_label_len], tf.float32)
        record = tf.io.parse_single_example(record, name_to_features)
        return record

    path = sys.argv[1]
    dataset = tf.data.TFRecordDataset(path)

    for item in dataset:
        try:
            record = decode_record(item)
        except Exception as e:
            print(e)
            print(item)
            break



    return NotImplemented


if __name__ == "__main__":
    main()