from data_generator.text_encoder import SubwordTextEncoder, CLS_ID, SEP_ID
from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair
import tensorflow as tf
import csv
from path import data_path
from cache import *
from evaluation import *
from collections import Counter
import unicodedata
num_classes = 2

corpus_dir = os.path.join(data_path, "nli")




class DataLoader:
    def __init__(self, max_sequence, vocab_filename, using_alt_tokenizer= False):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.train_file = os.path.join(corpus_dir, "train.tsv")
        self.dev_file = os.path.join(corpus_dir, "dev.tsv")
        self.max_seq = max_sequence
        voca_path = os.path.join(data_path, vocab_filename)
        assert os.path.exists(voca_path)
        self.name = "rte"
        if not using_alt_tokenizer:
            self.encoder = SubwordTextEncoder(voca_path)
            self.sep_char = "_"
            self.lower_case = False
        else:
            self.lower_case = True
            self.sep_char = "#"
            self.encoder = FullTokenizerWarpper(voca_path)


    def get_train_data(self):
        if self.train_data is None:
            self.train_data = load_cache("rte_train_cache")

        if self.train_data is None:
            self.train_data = list(self.example_generator(self.train_file))
        save_to_pickle(self.train_data, "rte_train_cache")
        return self.train_data

    def get_dev_data(self):
        if self.dev_data is None:
            self.dev_data = load_cache("rte_dev_cache")

        if self.dev_data is None:
            self.dev_data = list(self.example_generator(self.dev_file))
        save_to_pickle(self.dev_data, "rte_dev_cache")
        return self.dev_data

    def class_labels(self):
        return ["entailment", "not_entailment",]

    def example_generator(self, filename):
        label_list = self.class_labels()
        for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Works for both splits even though dev has some extra human labels.
            s1, s2 = split_line[1:3]
            l = label_list.index(split_line[-1])
            entry = self.encode(s1, s2)

            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"], l

    def encode(self, s1, s2):
        return self.encode_pair(s1, s2)

    def encode_pair(self, text_a, text_b):
        tokens_a = self.encoder.encode(text_a)
        tokens_b = self.encoder.encode(text_b)

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq - 3)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append(CLS_ID)
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(SEP_ID)
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append(SEP_ID)
            segment_ids.append(1)

        input_ids = tokens

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq
        assert len(input_mask) == self.max_seq
        assert len(segment_ids) == self.max_seq

        return {
            "input_ids": input_ids,
            "input_mask":input_mask,
            "segment_ids": segment_ids
        }
