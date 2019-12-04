from cache import *
from data_generator.common import get_tokenizer
from dictionary.reader import DictionaryReader
from tlm.dictionary.data_gen import dictionary_encoder


def encode_dictionary():
    path = "c:\\work\\Data\\webster\\webster_headerless.txt"
    d1 = DictionaryReader.open(path)
    d = dictionary_encoder(d1.entries, get_tokenizer())
    save_to_pickle(d, "webster")

