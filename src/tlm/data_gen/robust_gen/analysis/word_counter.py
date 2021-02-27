import sys

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import get_dir_files, Averager


def main():
    dir_path = sys.argv[1]
    tokenizer = get_tokenizer()
    averager = Averager()

    for file_path in get_dir_files(dir_path):
        for idx, record in enumerate(tf.compat.v1.python_io.tf_record_iterator(file_path)):
            if idx % 3 :
                continue
            example = tf.train.Example()
            example.ParseFromString(record)
            feature = example.features.feature
            input_mask = feature["input_mask"].int64_list.value
            if input_mask[-1]:
                input_ids = feature["input_ids"].int64_list.value
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                sep_idx1 = tokens.index("[SEP]")
                sep_idx2 = tokens.index("[SEP]", sep_idx1+1)
                doc_tokens = tokens[sep_idx1:sep_idx2]
                continue_cnt = 0
                for t in doc_tokens:
                    if t[:2] == "##":
                        continue_cnt += 1
##
                n_words = len(doc_tokens) - continue_cnt
                averager.append(n_words)

    print("average", averager.get_average())




if __name__ == "__main__":
    main()