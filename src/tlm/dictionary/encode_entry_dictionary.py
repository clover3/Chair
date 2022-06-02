from cache import *
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.dictionary.data_gen import dictionary_encoder
from tlm.dictionary.reader import DictionaryReader


def encode_dictionary():
    path = "c:\\work\\Data\\webster\\webster_headerless.txt"
    d1 = DictionaryReader.open(path)
    d = dictionary_encoder(d1.entries, get_tokenizer())
    save_to_pickle(d, "webster")

