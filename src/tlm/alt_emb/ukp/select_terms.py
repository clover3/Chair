from cache import save_to_pickle, load_from_pickle
from cpath import pjoin, data_path
from data_generator.argmining.ukp_header import all_topics
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.alt_emb.add_alt_emb import selected_terms_to_match_tree
from tlm.alt_emb.select_words import get_continuation_token_ids, build_word_tf


def count_tf():
    continuation_tokens = get_continuation_token_ids()

    dataset_dir = pjoin(data_path, "ukp_300")
    for topic in all_topics:
        train_data_path = pjoin(dataset_dir, "train_{}".format(topic))
        test_data_path = pjoin(dataset_dir, "dev_{}".format(topic))
        tf_train = build_word_tf(continuation_tokens, train_data_path)
        tf_dev = build_word_tf(continuation_tokens, test_data_path)

        save_to_pickle(tf_train, "tf_train_{}".format(topic))
        save_to_pickle(tf_dev, "tf_dev_{}".format(topic))


def convert_from_ids_to_words(tokenizer, most_common_itr):
    for word, cnt in most_common_itr:
        token_ids = list([int(t) for t in word.split()])
        tokens = list([tokenizer.inv_vocab[t] for t in token_ids])
        yield token_ids, tokens


def select_terms():
    words = []
    tokenizer = get_tokenizer()
    for topic in all_topics:
        tf_train = load_from_pickle("tf_train_{}".format(topic))
        tf_dev = load_from_pickle("tf_dev_{}".format(topic))

        selected_words = list(convert_from_ids_to_words(tokenizer, tf_dev.most_common(100)))
        print(selected_words)
        words.extend(selected_words)
    save_to_pickle(words, "ukp_selected")


def build_match_tree():
    selected_words = load_from_pickle("ukp_selected")
    match_tree = selected_terms_to_match_tree(selected_words)
    save_to_pickle(match_tree, "match_tree_ukp")


if __name__ == "__main__":
    build_match_tree()
