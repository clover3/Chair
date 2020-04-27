

# Select words from dev_nli that appears often in the dev set but does not appear in training set.

from base_type import FileName
from cache import save_to_pickle, load_from_pickle
from cpath import output_path, pjoin
from data_generator.common import get_tokenizer
from list_lib import left, lmap
from tlm.alt_emb.add_alt_emb import MatchTree, MatchNode
from tlm.alt_emb.select_words import get_continuation_token_ids, build_word_tf, select_new_words, select_common


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

    selected_words = select_new_words(tf_dev, tf_train)
    save_to_pickle(selected_words, "nli_selected_words")


def select_word_from_dev():
    tokenizer = get_tokenizer()

    tf_dev = load_from_pickle("nli_tf_dev_mis")
    selected_words = select_common(tf_dev, tokenizer)

    print(list(tf_dev.most_common(100))[-1])

    save_to_pickle(selected_words, "nli_dev_selected_words")


def show_common_words():
    tf_train = load_from_pickle("nli_tf_train")

    def decode_word(word):
        tokens = [tokenizer.inv_vocab[int(t)] for t in word.split()]
        return "".join(tokens)

    tokenizer = get_tokenizer()

    commons = list(tf_train.most_common(1000))
    common_terms = left(commons)
    common_terms = lmap(decode_word, common_terms)
    for st in [0, 100, 200, 300, 400]:
        print(commons[st])
        print(common_terms[st: st + 100])




def build_match_tree():
    selected_words = load_from_pickle("nli_dev_selected_words")

    seq_set = left(selected_words)

    match_tree = MatchTree()
    for seq in seq_set:
        match_tree.add_seq(seq)

    save_to_pickle(match_tree, "match_tree_nli_dev")


def build_debug_match_tree():
    match_tree = MatchTree()
    match_tree.add_seq([1997, 4597])
    save_to_pickle(match_tree, "debug_match_tree_nli")


def show_match_tree():
    match_tree:MatchTree = load_from_pickle("match_tree_nli_dev")

    def travel(node: MatchNode):
        print("tokens: ", node.token)
        print("childs: ", node.child_keys)
        if node.is_end:
            print("new_ids:", node.new_ids)
        for key, value in node.child_dict.items():
            travel(value)

    travel(match_tree.root)


if __name__ == "__main__":
    show_match_tree()
