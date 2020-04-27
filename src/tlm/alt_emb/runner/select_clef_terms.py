

# Select words from dev_nli that appears often in the dev set but does not appear in training set.

from base_type import FileName
from cache import save_to_pickle, load_from_pickle
from cpath import output_path, pjoin
from data_generator.common import get_tokenizer
from list_lib import left, lmap
from tlm.alt_emb.add_alt_emb import MatchTree
from tlm.alt_emb.select_words import get_continuation_token_ids, build_word_tf, select_common

def count_tf():
    continuation_tokens = get_continuation_token_ids()

    dir_path = pjoin(output_path, FileName("eHealth"))

    tf_train = build_word_tf(continuation_tokens, pjoin(dir_path, FileName("tfrecord_train")))
    tf_dev = build_word_tf(continuation_tokens, pjoin(dir_path, FileName("tfrecord_test")))

    save_to_pickle(tf_train, "clef1_tf_train")
    save_to_pickle(tf_dev, "clef1_tf_test")


def show_common_words():
    tf_train = load_from_pickle("clef1_tf_test")

    def decode_word(word):
        tokens = [tokenizer.inv_vocab[int(t)] for t in word.split()]
        return "".join(tokens)

    tokenizer = get_tokenizer()

    commons = list(tf_train.most_common(3000))
    common_terms = left(commons)
    common_terms = lmap(decode_word, common_terms)
    for st in [0, 200, 400, 600, 2800]:
        print(commons[st])
        print(common_terms[st: st + 100])


def select_word_from_test():
    tokenizer = get_tokenizer()

    tf_dev = load_from_pickle("clef1_tf_test")
    selected_words = select_common(tf_dev, tokenizer)

    print(list(tf_dev.most_common(3000))[-1])

    save_to_pickle(selected_words, "clef1_tf_test")


def build_match_tree():
    selected_words = load_from_pickle("clef1_tf_test")

    seq_set = left(selected_words)

    match_tree = MatchTree()
    for seq in seq_set:
        match_tree.add_seq(seq)

    save_to_pickle(match_tree, "match_tree_clef1_test")


if __name__ == "__main__":
    build_match_tree()
