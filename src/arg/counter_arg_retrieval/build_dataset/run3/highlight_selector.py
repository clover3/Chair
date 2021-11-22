from typing import List

from arg.perspectives.pc_tokenizer import PCTokenizerEx
from models.classic.stemming import StemmedToken
from models.classic.stopword import load_stopwords_for_query


class HighlightSelector:
    def __init__(self):
        self.stopwords = load_stopwords_for_query()
        self.pc_tokenizer_ex = PCTokenizerEx(on_unicode_error="keep")

    def split_stem_text(self, text) -> List[StemmedToken]:
        return self.pc_tokenizer_ex.tokenize_stem(text)

    def stem_tokens(self, tokens) -> List[StemmedToken]:
        def enc(token):
            try:
                t_out = self.pc_tokenizer_ex.stemmer.stem(token)
            except UnicodeDecodeError:
                t_out = token
            return StemmedToken(tokens, t_out)
        return list(map(enc, tokens))

    def get_highlight_indices_inner(self, q_tokens: List[StemmedToken], d_tokens: List[StemmedToken]):
        q_tokens_set = {s.stemmed_token for s in q_tokens}
        out_indices = []
        matched_stopwords = []
        for idx, token in enumerate(d_tokens):
            is_matching = token.stemmed_token in q_tokens_set
            is_stopwords = token.stemmed_token in self.stopwords

            if is_matching:
                if not is_stopwords:
                    out_indices.append(idx)
                else: # if stopwords
                    matched_stopwords.append(idx)

        def token_at(idx):
            try:
                return d_tokens[idx]
            except IndexError:
                return "Out of Index"

        retry = True
        n_loop = 0
        while retry:
            n_loop += 1
            if n_loop > 100:
                raise Exception
            might_cascade = False
            last_out_indices_len = len(out_indices)
            for idx in matched_stopwords:
                if idx-1 in out_indices or idx+1 in out_indices:
                    out_indices.append(idx)
                elif token_at(idx-1) in self.stopwords or token_at(idx+1) in self.stopwords:
                    might_cascade = True

            any_increase = len(out_indices) > last_out_indices_len
            retry = might_cascade and any_increase
        out_indices.sort()
        return out_indices