import os

from cpath import data_path
from data_generator.data_parser import load_protest
from data_generator.tokenizer_b import FullTokenizerWarpper, EncoderUnit


class DataLoader:
    def __init__(self, max_sequence, vocab_filename, voca_size):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        voca_path = os.path.join(data_path, vocab_filename)
        assert os.path.exists(voca_path)
        print(voca_path)


        self.lower_case = True
        self.sep_char = "#"
        self.encoder = FullTokenizerWarpper(voca_path)
        self.voca_size = voca_size
        self.dev_explain = None
        self.encoder_unit = EncoderUnit(max_sequence, voca_path)

    def get_train_data(self):
        if self.train_data is None:
            self.train_data = list(self.example_generator("train"))
        return self.train_data

    def get_dev_data(self):
        if self.dev_data is None:
            self.dev_data = list(self.example_generator("dev"))
        return self.dev_data

    def example_generator(self, split_name):
        X, Y = load_protest.load_data(split_name)
        for idx, x in enumerate(X):
            name, text = x
            l = Y[name]
            entry = self.encode(text)
            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"], l


    def encode(self, text):
        tokens_a = self.encoder.encode(text)
        return self.encoder_unit.encode_inner(tokens_a, [])

