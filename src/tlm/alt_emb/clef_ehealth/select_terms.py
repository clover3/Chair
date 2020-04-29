from cache import save_to_pickle, load_from_pickle
from cpath import pjoin, output_path
from tlm.alt_emb.select_words import get_continuation_token_ids, build_word_tf
from tlm.alt_emb.select_words import select_new_words


def count_tf():
    continuation_tokens = get_continuation_token_ids()

    out_dir = pjoin(output_path, "eHealth")
    train_save_path = pjoin(out_dir, "tfrecord_train")
    test_save_path = pjoin(out_dir, "tfrecord_test")
    tf_train = build_word_tf(continuation_tokens, train_save_path)
    tf_dev = build_word_tf(continuation_tokens, test_save_path)

    save_to_pickle(tf_dev, "eHealth_tf_train")
    save_to_pickle(tf_train, "eHealth_tf_dev")


def select_terms():
    tf_dev = load_from_pickle("eHealth_tf_train")
    tf_train = load_from_pickle("eHealth_tf_dev")

    selected_words = select_new_words(tf_dev, tf_train)
    save_to_pickle(selected_words, "eHealth_selected")


if __name__ == "__main__":
    select_terms()
