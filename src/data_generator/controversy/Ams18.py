from cpath import data_path
from data_generator.data_parser import amsterdam
from data_generator.tokenizer_b import EncoderUnit
from list_lib import lmap
from misc_lib import *


class DataLoader:
    def __init__(self, max_sequence, vocab_filename, voca_size):
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.source_collection = amsterdam

        voca_path = os.path.join(data_path, vocab_filename)
        assert os.path.exists(voca_path)
        print(voca_path)


        self.lower_case = True
        self.sep_char = "#"
        self.voca_size = voca_size
        self.dev_explain = None
        self.max_sequence = max_sequence
        self.voca_path = voca_path

    def get_train_data(self):
        if self.train_data is None:
            self.train_data = list(self.example_generator("train"))
            random.shuffle(self.train_data)
        return self.train_data

    def get_dev_data(self):
        if self.dev_data is None:
            self.dev_data = list(self.example_generator("val"))
            random.shuffle(self.dev_data)
        return self.dev_data

    @staticmethod
    def encode_entries(entries, encoder_unit):
        def enc(e):
            text = e['title'] + "\t" + e['content']
            l = e["label"]
            entry = encoder_unit.encode_text_single(text)
            return entry["input_ids"], entry["input_mask"], entry["segment_ids"], l
        return lmap(enc, entries)


    def example_generator(self, split_name):
        entries = amsterdam.load_data_split(split_name)
        print(split_name, "{} items".format(len(entries)))
        encoder_unit = EncoderUnit(self.max_sequence, self.voca_path)
        if len(entries) > 200:
            fn = lambda x: self.encode_entries(x, encoder_unit)
            return parallel_run(entries, fn, 10)
        else:
            return self.encode_entries(entries, encoder_unit)


    def encode_docs(self, docs):
        encoder_unit = EncoderUnit(self.max_sequence, self.voca_path)

        def enc(text):
            entry = encoder_unit.encode_text_single(text)
            return entry["input_ids"], entry["input_mask"], entry["segment_ids"]
        return lmap(enc, docs)
