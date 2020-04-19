

# Select words from dev_nli that appears often in the dev set but does not appear in training set.
from collections import Counter

from base_type import FileName
from cache import save_to_pickle, load_from_pickle
from cpath import output_path, pjoin
from data_generator.common import get_tokenizer
from list_lib import left
from tlm.alt_emb.add_alt_emb import MatchTree, MatchNode
from tlm.alt_emb.select_words import get_continuation_token_ids, build_word_tf


def count_tf():
    continuation_tokens = get_continuation_token_ids()

    dir_path = pjoin(output_path, FileName("nli_tfrecord_cls_300"))

    tf_train = build_word_tf(continuation_tokens, pjoin(dir_path, FileName("train")))
    tf_dev = build_word_tf(continuation_tokens, pjoin(dir_path, FileName("dev_mis")))

    save_to_pickle(tf_dev, "nli_tf_dev_mis")
    save_to_pickle(tf_train, "nli_tf_train")


def select_words():
    tf_dev = load_from_pickle("nli_tf_dev_mis")
    tf_train = load_from_pickle("nli_tf_train")

    print("train #words: ", len(tf_train))
    print("dev #words: ", len(tf_dev))

    tf_dev_new = Counter()
    for key in tf_dev:
        if key not in tf_train:
            tf_dev_new[key] = tf_dev[key]

    tokenizer = get_tokenizer()
    print("dev-train #words: ", len(tf_dev_new))
    for word, cnt in tf_dev_new.most_common(100):
        tokens = [tokenizer.inv_vocab[int(t)] for t in word.split()]
        print(tokens, cnt)

    selected_words = []
    for word, cnt in tf_dev_new.most_common(100):
        token_ids = list([int(t) for t in word.split()])
        tokens = list([tokenizer.inv_vocab[t] for t in token_ids])
        selected_words.append((token_ids, tokens))

    save_to_pickle(selected_words, "nli_selected_words")


def build_match_tree():
    selected_words = load_from_pickle("nli_selected_words")

    seq_set = left(selected_words)

    match_tree = MatchTree()
    for seq in seq_set:
        match_tree.add_seq(seq)

    save_to_pickle(match_tree, "match_tree_nli")


def build_debug_match_tree():
    match_tree = MatchTree()
    match_tree.add_seq([1997, 4597])
    save_to_pickle(match_tree, "debug_match_tree_nli")


def show_match_tree():
    match_tree:MatchTree = load_from_pickle("match_tree_nli")

    def travel(node: MatchNode):
        print("tokens: ", node.token)
        print("childs: ", node.child_keys)
        if node.is_end:
            print("new_ids:", node.new_ids)
        for key, value in node.child_dict.items():
            travel(value)

    travel(match_tree.root)


if __name__ == "__main__":
    build_debug_match_tree()
