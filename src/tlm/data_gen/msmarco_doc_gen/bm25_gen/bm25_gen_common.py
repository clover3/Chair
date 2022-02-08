from collections import Counter
from typing import List, Tuple, NamedTuple

from arg.bm25 import BM25
from cache import load_from_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lflatten
from misc_lib import find_max_idx
from tlm.data_gen.adhoc_datagen import get_combined_tokens_segment_ids
from tlm.data_gen.doc_encode_common import draw_window_size


class PassageSegment(NamedTuple):
    bert_tokens: List[str]
    bm25_tokens: List[List[str]]


class DocRep(NamedTuple):
    qid: str
    doc_id: str
    passage_segment_list: List[PassageSegment]
    label: str


class LongSentenceError(Exception):
    pass


def count_space_tokens(tokens):
    cnt = 0
    for t in tokens:
        if not t[0] == "#":
            cnt += 1
    return cnt


class BM25SelectedSegmentEncoder:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.bm25_scorer = get_bm25_scorer_for_mmd()
        self.passage_segment_enumerator = PassageSegmentEnumerator(max_seq_length)
        self.bert_tokenizer = get_tokenizer()

    def encode(self,
                 query_text: str,
                 bert_title_tokens: List[str],
                 bert_doc_tokens: List[List[str]],
                 bm25_title_tokens: List[str],
                 bm25_doc_tokens: List[List[str]],
                 no_short=False
                 ) -> List[Tuple[List, List]]:
        q_tokens: List[str] = self.bert_tokenizer.tokenize(query_text)
        q_tokens_for_bm25: List[str] = self.bm25_scorer.tokenizer.tokenize_stem(query_text)
        q_tf = Counter(q_tokens_for_bm25)
        segments: List[PassageSegment] = self.passage_segment_enumerator.encode(
            q_tokens, bert_title_tokens, bert_doc_tokens,
            bm25_title_tokens, bm25_doc_tokens, no_short)


        def score(segment: PassageSegment) -> float:
            body_tokens = lflatten(segment.bm25_tokens)
            body_tf = Counter(bm25_title_tokens + body_tokens)
            return self.bm25_scorer.score_inner(q_tf, body_tf)

        if segments:
            max_idx = find_max_idx(score, segments)
            best_segment = segments[max_idx]

            if len(best_segment) > 0:
                assert type(best_segment.bert_tokens[0]) == str
            cut_len = self.passage_segment_enumerator.title_token_max
            bert_title_tokens = bert_title_tokens[:cut_len]
            second_tokens = bert_title_tokens + best_segment.bert_tokens
        else:
            second_tokens = []
        tokens, segment_ids = get_combined_tokens_segment_ids(q_tokens, second_tokens)
        assert len(tokens) <= self.max_seq_length
        return [(tokens, segment_ids)]


def get_bm25_scorer_for_mmd() -> BM25:
    df = load_from_pickle("mmd_df_10")
    avdl_raw = 1350
    avdl_passage = 40

    k1 = 0.1
    max_seq_length = 512
    bm25 = BM25(df, avdl=avdl_passage, num_doc=321384, k1=k1, k2=100, b=0.75)
    return bm25


class TokensListPooler:
    def __init__(self, bert_tokens_list: List[List[str]], bm25_tokens_list: List[List[str]]):
        self.bert_tokens_list = bert_tokens_list
        self.bm25_tokens_list = bm25_tokens_list
        self.cursor = 0
        self.remaining_tokens = []

    def pool(self, max_acceptable_size, slice_if_too_long) -> Tuple[List[str], List[str]]:
        if not self.remaining_tokens:
            cur_bert_tokens: List[str] = self.bert_tokens_list[self.cursor]
            cur_bm25_tokens: List[str] = self.bm25_tokens_list[self.cursor]
            assert type(cur_bm25_tokens) == list
        else:
            cur_bert_tokens, cur_bm25_tokens = self.remaining_tokens

        if len(cur_bert_tokens) <= max_acceptable_size:
            if self.remaining_tokens:
                self.remaining_tokens = []
            self.cursor += 1
            return cur_bert_tokens, cur_bm25_tokens
        elif slice_if_too_long:
            head_bert = cur_bert_tokens[:max_acceptable_size]
            tail_bert = cur_bert_tokens[max_acceptable_size:]
            assert type(cur_bm25_tokens) == list
            maybe_bm25_token_len = count_space_tokens(head_bert)
            head_bm25 = cur_bm25_tokens[:maybe_bm25_token_len]
            tail_bm25 = cur_bm25_tokens[maybe_bm25_token_len:]
            self.remaining_tokens = tail_bert, tail_bm25
            return head_bert, head_bm25
        else:
            return [], []

    def is_end(self):
        return self.cursor >= len(self.bert_tokens_list)


class PassageSegmentEnumerator:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.tokenizer = get_tokenizer()
        self.title_token_max = 64
        self.query_token_max = 64

    def encode(self,
                 q_tokens,
               bert_title_tokens: List[str],
               bert_doc_tokens: List[List[str]],
               bm25_title_tokens: List[str],
               bm25_doc_tokens: List[List[str]],
               no_short=False
                 ) -> List[PassageSegment]:
        doc_align_map = None
        if len(bm25_doc_tokens) != len(bert_doc_tokens):
            assert False

        def clip_length(bert_tokens, space_tokens, n_max_tokens):
            if len(bert_tokens) > n_max_tokens:
                return count_space_tokens(bert_tokens)
            else:
                return len(space_tokens)

        clipped_title_len = clip_length(bert_title_tokens, bm25_title_tokens, self.title_token_max)
        bm25_title_tokens = bm25_title_tokens[:clipped_title_len]
        bert_title_tokens = bert_title_tokens[:self.title_token_max]

        q_tokens = q_tokens[:self.query_token_max]
        content_len = self.max_seq_length - 3 - len(q_tokens)
        window_size = content_len
        passage_segment_list: List[PassageSegment] = []

        if len(bm25_doc_tokens) :
            if not type(bm25_doc_tokens[0]) == list:
                print(type(bm25_doc_tokens[0]))
                print(bm25_doc_tokens)
                assert False
            assert type(bm25_doc_tokens[0][0]) == str

        pooler = TokensListPooler(bert_doc_tokens, bm25_doc_tokens)
        cnt = 0
        while not pooler.is_end():
            bert_tokens: List[str] = []
            bm25_tokens: List[List[str]] = []
            n_sent = 0
            if no_short:
                current_window_size = window_size
            else:
                current_window_size = draw_window_size(window_size)
            maybe_max_body_len = current_window_size - len(bert_title_tokens)

            new_bert_tokens, new_bm25_tokens = pooler.pool(maybe_max_body_len, n_sent == 0)
            n_sent += 1

            bert_tokens.extend(new_bert_tokens)
            bm25_tokens.append(new_bm25_tokens)
            cnt += 1
            while not pooler.is_end():
                cnt += 1
                available_length = maybe_max_body_len - len(bert_tokens)
                new_bert_tokens, new_bm25_tokens = pooler.pool(available_length, n_sent == 0)
                if not new_bert_tokens:
                    break
                bert_tokens.extend(new_bert_tokens)
                bm25_tokens.append(new_bm25_tokens)
                assert cnt < 10000
            ps = PassageSegment(bert_tokens, bm25_tokens)
            passage_segment_list.append(ps)
        return passage_segment_list
