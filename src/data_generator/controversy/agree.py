import os
from path import data_path
from data_generator.data_parser import ukp
from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair
from data_generator.text_encoder import SubwordTextEncoder, CLS_ID, SEP_ID
from models.classic.stopword import load_stopwords
from collections import Counter
from misc_lib import *
from data_generator.data_parser import agree

class DataLoader:
    def __init__(self, max_sequence, vocab_filename):
        self.max_seq = max_sequence
        voca_path = os.path.join(data_path, vocab_filename)
        self.lower_case = True
        self.sep_char = "#"
        self.encoder = FullTokenizerWarpper(voca_path)
        self.all_data = list(agree.load_agree_data())
        self.dev_data = self.all_data[:200]
        self.train_data = self.all_data[200:]

    def gen_data(self, data):
        result = []
        for x, y in data:
            entry = self.encode(x)
            e = entry["input_ids"], entry["input_mask"], entry["segment_ids"], y
            result.append(e)
        return result

    def get_train_data(self):
        return self.gen_data(self.train_data)

    def get_dev_data(self):
        return self.gen_data(self.dev_data)

    def encode(self, text_a):
        tokens_a = self.encoder.encode(text_a)
        tokens_b = []

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



if __name__ == "__main__":
    d = DataLoader(200, "bert_voca.txt")
    d.get_train_data()
    d.get_dev_data()
