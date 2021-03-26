import os

from cpath import data_path
from data_generator.NLI.nli import DataLoader
from data_generator.tfrecord_gen import modify_data_loader


def get_bio_nli_voca_path():
    return os.path.join(data_path, "biobert_voca.txt")


def get_biobert_nli_data_loader(sequence_length):
    data_loader = DataLoader(sequence_length, "biobert_voca.txt", True, True)
    return modify_data_loader(data_loader)

