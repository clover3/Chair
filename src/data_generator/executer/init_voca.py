import tensorflow as tf

import itertools
from data_generator.stance import stance_detection
from data_generator.mask_lm.wiki_lm import train_path
from data_generator.generator_utils import get_or_generate_vocab_inner
from data_generator.common import data_path
def file_sampler(filepath, file_byte_budget=1e6):
    with tf.gfile.GFile(filepath, mode="r") as source_file:
        file_byte_budget_ = file_byte_budget
        counter = 0
        countermax = int(source_file.size() / file_byte_budget_ / 2)
        for line in source_file:
            if counter < countermax:
                counter += 1
            else:
                if file_byte_budget_ <= 0:
                    break
                line = line.strip()
                file_byte_budget_ -= len(line)
                counter = 0
                yield line

vocab_filename = "shared_voca.txt"

def init_shared_voca():
    vocab_size = 32000
    stance_text = stance_detection.get_train_text()
    wiki_text = file_sampler(train_path)

    vocab_generator = itertools.chain(stance_text, wiki_text)
    return get_or_generate_vocab_inner(data_path, vocab_filename, vocab_size,
                                       vocab_generator)


