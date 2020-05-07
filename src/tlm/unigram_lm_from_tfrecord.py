from collections import Counter

from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_util.enum_features import load_record


class LM:
    def __init__(self, as_subword, tokenizer=None):
        self.as_subword = as_subword
        if not as_subword:
            self.continuation = set()
            self.inv_vocab = tokenizer.inv_vocab
            assert tokenizer is not None
            for token_id, subword in tokenizer.inv_vocab.items():
                if subword[:2] == "##":
                    self.continuation.add(token_id)

        self.tf = Counter()

    def update(self, input_ids):
        if self.as_subword:
            self.tf.update(list(input_ids))
        else:
            words = self.get_words(list(input_ids))
            self.tf.update(words)

    def get_words(self, input_ids):
        words = []
        cur_word = []
        for t in input_ids:
            w = self.inv_vocab[t]
            if t in self.continuation:
                cur_word.append(w)
            else:
                words.append(cur_word)
                cur_word = [w]
        if cur_word:
            words.append(cur_word)

        for word in words:
            yield "".join(word)

    def update_from_lm(self, other_lm):
        for key in other_lm.tf:
            tf = other_lm.tf[key]
            self.tf[key] += tf


def get_lm_tf(fn, sample_size=None, as_subword=True):
    tokenizer = get_tokenizer()
    tfrecord_itr = load_record(fn)

    lm = LM(as_subword, tokenizer)

    for idx, inst in enumerate(tfrecord_itr):
        if sample_size is not None and idx > sample_size:
            break
        input_ids = inst["input_ids"].int64_list.value
        lm.update(input_ids)

    return lm.tf
