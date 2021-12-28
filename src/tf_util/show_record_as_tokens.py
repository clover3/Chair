import sys

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer


def file_show(fn):
    cnt = 0
    tokenizer = get_tokenizer()
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = feature.keys()

        print("---- record -----")
        for key in keys:
            if key == "masked_lm_weights":
                v = feature[key].float_list.value
            else:
                v = feature[key].int64_list.value

            print(key)
            print(v)

            if key in ["input_ids", "input_ids1", "input_ids2", "q_e_input_ids"]:
                tokens = tokenizer.convert_ids_to_tokens(v)
                print(key)
                print(" ".join(tokens))



        cnt += 1
        if cnt >= 5:
            break


if __name__ == "__main__":
    file_show(sys.argv[1])
