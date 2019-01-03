import tensorflow as tf
import os
import itertools
from data_generator.stance import stance_detection
from data_generator.mask_lm import enwiki
from data_generator.generator_utils import get_or_generate_vocab_inner
from data_generator.text_encoder import TokenTextEncoder
from data_generator.common import data_path
from data_generator.data_parser import tweet_reader
from data_generator.shared_setting import Tweets2Stance, TopicTweets2Stance, SimpleTokner
from data_generator.tokenizer import encode
from misc_lib import flatten
from data_generator.NLI import nli

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


def list_sampler(src_list, target_size = 1e6):
    counter = 0
    selected_size = 0
    countermax = int(len(src_list)/target_size)
    print(" Sampling {} from {}".format(target_size, len(src_list)))
    for e in src_list:
        if counter < countermax:
            counter += 1
        else:
            if selected_size >= target_size:
                break
            selected_size += 1
            counter = 0
            yield e



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
    topic = "hillary"
    setting = TopicTweets2Stance(topic)

    vocab_size = setting.vocab_size
    stance_text = stance_detection.get_train_text()
    sents = list(tweets.load_as_text_chunk(topic))
    aux_text = list_sampler(sents)
    out_vocab_filename = setting.vocab_filename
    vocab_generator = itertools.chain(stance_text, aux_text)
    return get_or_generate_vocab_inner(data_path, out_vocab_filename, vocab_size,
                                       vocab_generator)


def init_token_voca():
    topic = "atheism"
    setting = SimpleTokner(topic)

    stance_text = stance_detection.get_train_text()
    token_list = list([l.split() for l in stance_text])
    print(token_list[:20])
    encoder = TokenTextEncoder(None, vocab_list=flatten(token_list))
    encoder.store_to_file(setting.vocab_filename)


def init_voca_nli():
    vocab_size = 32000
    train_file = os.path.join(nli.corpus_dir, "train.tsv")

    def file_reader(filename):
        for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Works for both splits even though dev has some extra human labels.
            s1, s2 = split_line[8:10]
            yield s1
            yield s2
    vocab_generator = file_reader(train_file)
    out_vocab_filename = "NLI_voca.txt"
    return get_or_generate_vocab_inner(data_path, out_vocab_filename, vocab_size,
                                       vocab_generator)



if __name__ == "__main__":
    init_voca_nli()