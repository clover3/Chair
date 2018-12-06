import os
from data_generator.common import *
from data_generator.data_parser import tweet_reader
from data_generator.text_encoder import SubwordTextEncoder
from data_generator.shared_setting import Tweets2Stance
from data_generator.group_sampler import pos_neg_pair_sampling
from collections import Counter

def avg_token_length():
    s = "atheism"
    cont_list = tweet_reader.load_as_text_chunk(s)
    voca_path = os.path.join(data_path, Tweets2Stance.vocab_filename)

    encoder = SubwordTextEncoder(voca_path)


    n = 0
    histogram = Counter()
    for sent in cont_list:
        tokens = encoder.encode(sent)
        histogram[len(tokens)] += 1

        n += 1
        if n > 1000:
            break

    accum = 0
    for i in range(100):
        accum += histogram[i]
        print("{} : {}".format(i, accum))



def data_size():
    gd= tweet_reader.load_per_user("atheism")
    list(pos_neg_pair_sampling(gd))

