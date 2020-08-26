from data_generator.tokenizer_wo_tf import get_tokenizer


class SubwordConvertor:
    def __init__(self):
        self.continuation = set()
        tokenizer = get_tokenizer()
        self.inv_vocab = tokenizer.inv_vocab
        assert tokenizer is not None
        for token_id, subword in tokenizer.inv_vocab.items():
            if subword[:2] == "##":
                self.continuation.add(token_id)

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
