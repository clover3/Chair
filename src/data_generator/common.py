
import os
import pickle
import re
from os.path import dirname
import tensorflow as tf
from data_generator import text_encoder
from path import project_root, data_path

from data_generator import tokenizer

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

