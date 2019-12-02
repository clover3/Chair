import tensorflow as tf

from misc_lib import get_dir_files_sorted_by_mtime


def check_data(fn):
    raw_len = {
        "ab_mapping": 320,
        "d_input_ids":30720,
        "d_input_mask":30720,
        "d_location_ids":512,
        "d_segment_ids":30720,
        "input_ids":16384,
        "input_mask":16384,
        "masked_lm_ids":640,
        "masked_lm_positions":640,
        "segment_ids":16384,
        "selected_word":256
    }

    def get_len(feature, key):
        if key == "masked_lm_weights":
            v = feature[key].float_list.value
        else:
            v = feature[key].int64_list.value
        return len(v)

    print(fn)
    data_idx = 0
    for record in tf.python_io.tf_record_iterator(fn):

        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = list(feature.keys())

        all_data_len = list([(key, get_len(feature, key)) for key in feature])

        for key in raw_len:
            assert key in keys
            data_len = get_len(feature, key)
            if not data_len == raw_len[key]:
                print(all_data_len)
                raise Exception("Key {} len {} != {} at data_idx {}".format(key, data_len, raw_len[key], data_idx))

        data_idx += 1



def main():
    for file_path in get_dir_files_sorted_by_mtime("/mnt/nfs/work3/youngwookim/data/bert_tf/ssdr"):
        check_data(file_path)


if __name__ == "__main__":
    main()