import tensorflow as tf

import itertools
from data_generator.stance import stance_detection
from data_generator.mask_lm import enwiki
from data_generator.generator_utils import get_or_generate_vocab_inner
from data_generator.common import data_path
from data_generator.data_parser import tweets
from data_generator.shared_setting import Tweets2Stance

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



def init_voca_stance_n_enwiki():
    vocab_size = 32000
    stance_text = stance_detection.get_train_text()
    wiki_text = file_sampler(enwiki.train_path)
    out_vocab_filename = "shared_voca.txt"
    vocab_generator = itertools.chain(stance_text, wiki_text)
    return get_or_generate_vocab_inner(data_path, out_vocab_filename, vocab_size,
                                       vocab_generator)



def init_voca_stance_n_guardian():
    vocab_size = 32000
    stance_text = stance_detection.get_train_text()
    aux_text = NotImplemented
    out_vocab_filename = "voca_guardian.txt"
    vocab_generator = itertools.chain(stance_text, aux_text)
    return get_or_generate_vocab_inner(data_path, out_vocab_filename, vocab_size,
                                       vocab_generator)


def init_voca_stance_n_tweets():
    vocab_size = Tweets2Stance.vocab_size
    stance_text = stance_detection.get_train_text()
    aux_text = tweets.load_as_text_chunk("atheism")
    out_vocab_filename = Tweets2Stance.vocab_filename
    vocab_generator = itertools.chain(stance_text, aux_text)
    return get_or_generate_vocab_inner(data_path, out_vocab_filename, vocab_size,
                                       vocab_generator)





if __name__ == "__main__":
    init_voca_stance_n_tweets()