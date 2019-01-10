from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair, EncoderUnit
from data_generator.text_encoder import SubwordTextEncoder, TokenTextEncoder, CLS_ID, SEP_ID, EOS_ID

import tensorflow as tf
import csv
from path import data_path
from evaluation import *
num_classes = 3
import random
from data_generator.data_parser.trec import *
from rpc.dbpedia_server import TextReaderClient, read_dbpedia

scope_dir = os.path.join(data_path, "controversy")
corpus_dir = os.path.join(scope_dir, "mscore")



def read_mscore():
    path = os.path.join(corpus_dir, "MScore.txt")
    result = []
    for line in open(path, "r"):
        score, topic = line.split()
        score = float(score)
        result.append((topic, score))
    return result


def read_mscore_valid():
    return load_from_pickle("mscore_valid")
    mscore = read_mscore()
    new_mscore = []
    dbpedia = read_dbpedia()
    for doc_id, score in mscore:
        if doc_id not in dbpedia:
            None
        else:
            new_mscore.append((doc_id, score))

    print("Valid {} -> {}".format(len(mscore), len(new_mscore)))
    save_to_pickle(new_mscore, "mscore_valid")
    return new_mscore


class DataLoader:
    def __init__(self, max_sequence, vocab_filename, voca_size):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        voca_path = os.path.join(data_path, vocab_filename)
        assert os.path.exists(voca_path)
        print(voca_path)

        self.mscore = read_mscore_valid()
        self.mscore_dict = dict(self.mscore)
        self.train_topics, self.dev_topics = self.held_out(left(self.mscore))

        self.lower_case = True
        self.sep_char = "#"
        self.encoder = FullTokenizerWarpper(voca_path)
        self.voca_size = voca_size
        self.dev_explain = None
        self.encoder_unit = EncoderUnit(max_sequence, voca_path)
        self.client = TextReaderClient()

        class BiasSampler:
            def __init__(self, topics, score_dict):
                self.count = Counter()
                self.sample_space = []
                for topic in topics:
                    score = score_dict[topic]
                    if self.count[score] < 3:
                        self.sample_space.append(topic)
                        self.count[score] += 1

            # Sample from all group
            def sample(self):
                return random.sample(self.sample_space, 2)

        self.train_sampler = BiasSampler(self.train_topics, self.mscore_dict)
        self.dev_sampler = BiasSampler(self.dev_topics, self.mscore_dict)


    def get_train_data(self, size):
        return self.generate_data(self.train_sampler.sample, size)

    def get_dev_data(self, size):
        return self.generate_data(self.dev_sampler.sample, size)

    def generate_data(self, sample_fn, size):
        pair_n = int(size / 2)
        assert pair_n * 2 == size
        topic_pairs = self.sample_pairs(sample_fn, pair_n)
        result = []
        for topic_pair in topic_pairs:
            t1, t2 = topic_pair
            inst = (self.retrieve(t1),  self.retrieve(t2))
            result += list(self.encode_pair(inst))

        return result

    def retrieve(self, topic):
        r = self.client.retrieve(topic)
        if not r:
            print(topic)
        return r

    def encode_pair(self, sent_pair):
        for sent in sent_pair:
            entry = self.encode(sent)
            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"]

    def encode(self, text):
        tokens_a = self.encoder.encode(text)
        return self.encoder_unit.encode_inner(tokens_a, [])

    def sample_pairs(self, sample_fn, n_pairs):
        result = []
        for i in range(n_pairs):
            selected = sample_fn()
            t1 = selected[0]
            t2 = selected[1]
            score1 = self.mscore_dict[t1]
            score2 = self.mscore_dict[t2]
            if score1 < score2:
                result.append((t1, t2))
            else:
                result.append((t2, t1))
        return result

    def held_out(self,topics):
        heldout_size = int(len(topics) * 0.1)
        dev_topics = set(random.sample(topics, heldout_size))
        train_topics = set(topics) - dev_topics
        return train_topics, dev_topics


if __name__ == '__main__':
    mscore = read_mscore_valid()

