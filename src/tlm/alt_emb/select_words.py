



#  Read NLI train/dev files
#  Build word(multiple subwords) counter
#  Select word with highest TF that has zero TF in train
from collections import Counter
from typing import Set

from data_generator.common import get_tokenizer
from tf_util.enum_features import load_record_v2
from tlm.data_gen.feature_to_text import take


def is_real_example(feature):
    if "is_real_example" not in feature:
        return True
    return take(feature["is_real_example"])[0] == 1


def build_word_tf(continuation_tokens: Set[int], file_path):
    feature_itr = load_record_v2(file_path)
    counter = Counter()
    for feature in feature_itr:
        if not is_real_example(feature):
            continue

        input_ids = take(feature["input_ids"])
        cur_word = []
        for idx, token_id in enumerate(input_ids):
            if token_id in continuation_tokens:
                cur_word.append(token_id)
            else:
                if len(cur_word) > 1:
                    word_sig = " ".join([str(t) for t in cur_word])
                    counter[word_sig] += 1
                cur_word = [token_id]

    return counter


def get_continuation_token_ids() -> Set[int]:
    tokenizer = get_tokenizer()

    s = set()
    for token, token_id in tokenizer.vocab.items():
        if token[:2] == "##":
            s.add(token_id)
    return s


def select_new_words(tf_dev, tf_train):
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
    print(list(tf_dev_new.most_common(100))[-1])
    return selected_words


def select_common(tf_dev, tokenizer):
    selected_words = []
    for word, cnt in tf_dev.most_common(100):
        token_ids = list([int(t) for t in word.split()])
        tokens = list([tokenizer.inv_vocab[t] for t in token_ids])
        selected_words.append((token_ids, tokens))

    return selected_words