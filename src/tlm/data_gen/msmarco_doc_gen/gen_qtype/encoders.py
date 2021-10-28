from nltk import sent_tokenize
from typing import List, Callable, Tuple

from nltk import sent_tokenize

from data_generator.tokenizer_wo_tf import get_tokenizer, is_continuation
from list_lib import lmap
# Function Word Drop
from misc_lib import Averager
from tlm.data_gen.adhoc_sent_tokenize import enum_passage_random_short, enum_passage, join_tokens, split_by_window


class DropException(Exception):
    pass


class DropTokensDecider:
    def __init__(self, do_drop_fn: Callable[[str], bool], tokenizer):
        self.word_match_fn: Callable[[str], bool] = do_drop_fn
        self.tokenizer = tokenizer
        self.drop_rate_avg = Averager()

    def drop(self, tokens):
        length = len(tokens)

        def do_drop(index):
            token = tokens[index]
            if self.word_match_fn(token):
                is_last = (index + 1 == length)
                if is_last or not is_continuation(tokens[index+1]):
                    return True
                else:
                    return False
            else:
                return False

        out_tokens = []
        for j in range(length):
            if not do_drop(j):
                out_tokens.append(tokens[j])
        if not out_tokens:
            raise DropException("WARNING no tokens remain after filtering: {}".format(" ".join(tokens)))

        drop_rate = 1 - len(out_tokens) / len(tokens)
        self.drop_rate_avg.append(drop_rate)
        return out_tokens


class FromTextEncoderFWDrop:
    def __init__(self,
                 max_seq_length,
                 drop_word_match_fn,
                 random_short=False,
                 seg_selection_fn=None,
                 max_seg_per_doc=999999,
                 trim_len=64):
        self.max_seq_length = max_seq_length
        self.tokenizer = get_tokenizer()
        self.title_token_max = trim_len
        self.query_token_max = trim_len
        if random_short:
            self.enum_passage_fn = enum_passage_random_short
        else:
            self.enum_passage_fn = enum_passage

        self.seg_selection_fn = seg_selection_fn
        self.max_seg_per_doc = max_seg_per_doc

        self.decider = DropTokensDecider(drop_word_match_fn, self.tokenizer)
        self.drop_q_tokens: Callable[[List[str]], List[str]] = self.decider.drop

    def encode(self, query_tokens, title, body) -> List[Tuple[List, List, List, List]]:
        query_tokens = query_tokens[:self.query_token_max]
        query_tokens_dropped = self.drop_q_tokens(query_tokens)
        content_len = self.max_seq_length - 3 - len(query_tokens)
        assert content_len > 0

        title_tokens = self.tokenizer.tokenize(title)
        title_tokens = title_tokens[:self.title_token_max]

        maybe_max_body_len = content_len - len(title_tokens)
        body_tokens: List[List[str]] = self.get_tokens_sent_grouped(body, maybe_max_body_len)
        ##
        if not body_tokens:
            body_tokens: List[List[str]] = [["[PAD]"]]

        n_tokens = sum(map(len, body_tokens))
        insts = []
        for idx, second_tokens in enumerate(self.enum_passage_fn(title_tokens, body_tokens, content_len)):
            out_tokens, segment_ids = join_tokens(query_tokens, second_tokens)
            out_tokens_drop, segment_ids_drop = join_tokens(query_tokens_dropped, second_tokens)
            entry = out_tokens, segment_ids, out_tokens_drop, segment_ids_drop
            insts.append(entry)
            assert idx <= n_tokens
            if idx >= self.max_seg_per_doc:
                break

        if self.seg_selection_fn is not None:
            insts = self.seg_selection_fn(insts)

        return insts

    def get_tokens_sent_grouped(self, body, maybe_max_body_len) -> List[List[str]]:
        body_sents = sent_tokenize(body)
        body_tokens: List[List[str]] = lmap(self.tokenizer.tokenize, body_sents)
        body_tokens: List[List[str]] = list([tokens for tokens in body_tokens if tokens])
        out_body_tokens: List[List[str]] = []
        for tokens in body_tokens:
            if len(tokens) >= maybe_max_body_len:
                out_body_tokens.extend(split_by_window(tokens, maybe_max_body_len))
            else:
                out_body_tokens.append(tokens)

        return out_body_tokens

