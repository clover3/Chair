
import os

import tensorflow as tf

from cpath import data_path
from data_generator import text_encoder
from data_generator import tokenizer
from data_generator.tokenizer_b import FullTokenizerWarpper
from data_generator.tokenizer_wo_tf import EncoderUnitOld, FullTokenizer

_EXAMPLES_FILE = 'examples.txt'

# NOT USED / DELETE IT
def _get_or_generate_vocab(tmp_dir, vocab_filename, vocab_size):
    """Read or create vocabulary."""
    vocab_filepath = os.path.join(tmp_dir, vocab_filename)
    print('Vocab file written to: ' + vocab_filepath)

    if tf.gfile.Exists(vocab_filepath):
        gs = text_encoder.SubwordTextEncoder(vocab_filepath)
        return gs
    example_file = os.path.join(tmp_dir, _EXAMPLES_FILE)
    gs = text_encoder.SubwordTextEncoder()
    token_counts = tokenizer.corpus_token_counts(
        example_file, corpus_max_lines=1000000)
    gs = gs.build_to_target_size(
        vocab_size, token_counts, min_val=1, max_val=1e3)
    gs.store_to_file(vocab_filepath)
    return gs


def get_encoder():
    voca_path = os.path.join(data_path, "bert_voca.txt")
    return FullTokenizerWarpper(voca_path)


def get_encoder_unit(seq_length, modified=True):
    voca_path = os.path.join(data_path, "bert_voca.txt")
    encoder_unit = EncoderUnitOld(seq_length, voca_path)
    if modified:
        CLS_ID = encoder_unit.encoder.ft.convert_tokens_to_ids(["[CLS]"])[0]
        SEP_ID = encoder_unit.encoder.ft.convert_tokens_to_ids(["[SEP]"])[0]
        encoder_unit.CLS_ID = CLS_ID
        encoder_unit.SEP_ID = SEP_ID

    return encoder_unit


def get_tokenizer():
    voca_path = os.path.join(data_path, "bert_voca.txt")
    return FullTokenizer(voca_path)