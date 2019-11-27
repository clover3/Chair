from dictionary.reader import DictionaryReader
from tlm.dictionary.data_gen import dictionary_encoder
from data_generator.common import get_tokenizer
from cache import *
from sydney_manager import MarkedTaskManager
from tlm.dictionary.data_gen import DictTrainGen, Dictionary
from tlm.data_gen import run_unmasked_pair_gen
from tlm.tf_logging import tf_logging
import logging
from misc_lib import exist_or_mkdir



def encode_dictionary():
    path = "c:\\work\\Data\\webster\\webster_headerless.txt"
    d1 = DictionaryReader.open(path)
    d = dictionary_encoder(d1.entries, get_tokenizer())
    save_to_pickle(d, "webster")

