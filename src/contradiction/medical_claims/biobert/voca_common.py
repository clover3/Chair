import os

from cpath import data_path
from data_generator.tokenizer_wo_tf import FullTokenizer


def get_biobert_voca_path():
    return os.path.join(data_path, "biobert_voca.txt")


def get_biobert_tokenizer():
    return FullTokenizer(get_biobert_voca_path())