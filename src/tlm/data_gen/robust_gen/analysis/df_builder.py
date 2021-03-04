import sys
from collections import Counter

import tensorflow as tf

from cache import save_to_pickle
from data_generator.subword_convertor import SubwordConvertor
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from misc_lib import get_dir_files, Averager, TimeEstimator


def main():
    dir_path = sys.argv[1]
    tokenizer = get_tokenizer()
    averager = Averager()
    sbc = SubwordConvertor()
    df = Counter()
    collection_size = 0
    tikcer = TimeEstimator(485393)
    for file_path in get_dir_files(dir_path):
        for idx, record in enumerate(tf.compat.v1.python_io.tf_record_iterator(file_path)):
            example = tf.train.Example()
            example.ParseFromString(record)
            feature = example.features.feature
            input_ids = feature["input_ids"].int64_list.value
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            sep_idx1 = tokens.index("[SEP]")
            sep_idx2 = tokens.index("[SEP]", sep_idx1+1)
            doc_tokens = tokens[sep_idx1:sep_idx2]
            words = lmap(tuple, sbc.get_word_as_subtoken_tuple(doc_tokens))
            dl = len(words)
            collection_size += dl
            averager.append(dl)
            for word in set(words):
                df[word] += 1
            tikcer.tick()

    print("collection length", collection_size)
    print("average dl", averager.get_average())
    save_to_pickle(df, "subword_df_robust_train")




if __name__ == "__main__":
    main()