from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Tuple

from data_generator.data_parser.robust import load_robust_04_query
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten, list_join, lmap
from misc_lib import ceil_divide, get_second
from tlm.data_gen.classification_common import TokensAndSegmentIds
from tlm.data_gen.robust_gen.seg_lib.piece_score_parser import PieceScoreParser, Piece
from tlm.data_gen.robust_gen.seg_lib.seg_score_common import ScoredPieceFromPair
from tlm.robust.load import get_robust_qid_list


class Config1:
    n_factor = 16
    step_size = 16
    max_seq_length = 128
    max_seq_length2 = 128 - 16
    batch_size = 8


def make_second_piece_candidates(seg2_num_piece, sp_list: List[ScoredPieceFromPair], first_piece) -> List[Tuple[
    Piece, float]]:
    n_ss = len(sp_list)

    # Select second segment
    def get_score_at(idx):
        if first_piece.st <= idx < first_piece.ed:
            return 0
        else:
            return sp_list[idx].score

    second_piece_candidates = []
    for st_ss_idx in range(n_ss):
        ed_ss_idx = min(st_ss_idx + seg2_num_piece, n_ss)
        interval = Piece(st_ss_idx, ed_ss_idx)
        sensitivity_score = sum(get_score_at(idx) for idx in range(st_ss_idx, ed_ss_idx))
        score = sensitivity_score
        second_piece_candidates.append((interval, score))
    second_piece_candidates.sort(key=get_second, reverse=True)
    return second_piece_candidates


def make_first_piece_candidates(seg1_num_piece, sp_list) -> List[Tuple[Piece, float]]:
    n_ss = len(sp_list)
    first_piece_candidates = []
    for st_ss_idx in range(n_ss):
        st = sp_list[st_ss_idx].st
        ed_ss_idx = min(st_ss_idx + seg1_num_piece, n_ss)
        ed = sp_list[ed_ss_idx - 1].ed
        interval = Piece(st_ss_idx, ed_ss_idx)
        sensitivity_score = sum(sp_list[idx].score for idx in range(st_ss_idx, ed_ss_idx))
        location_score = 1 / (st_ss_idx + 1)
        score = sensitivity_score + location_score
        first_piece_candidates.append((interval, score))
    first_piece_candidates.sort(key=get_second, reverse=True)
    return first_piece_candidates


def select_a_two_piece_segment(probe_config, available_length, sp_list: List[ScoredPieceFromPair]) -> List[Piece]:
    seg1_num_piece = ceil_divide(ceil_divide(available_length, 2), probe_config.step_size)
    seg1_length = seg1_num_piece * probe_config.step_size
    # Select first segment

    first_piece_candidates: List[Tuple[Piece, float]] = make_first_piece_candidates(seg1_num_piece, sp_list)
    first_piece, _ = first_piece_candidates[0]

    seg2_length = available_length - seg1_length
    seg2_num_seg = int(seg2_length / probe_config.step_size)

    second_piece_candidates: List[Tuple[Piece, float]] = make_second_piece_candidates(seg2_num_seg, sp_list, first_piece)
    second_piece, _ = second_piece_candidates[0]

    return rearrange_interval(first_piece, second_piece)


def select_many_two_piece_segment(probe_config, available_length, sp_list: List[ScoredPieceFromPair], n1, n2)\
        -> Iterable[List[Piece]]:
    seg1_num_piece = ceil_divide(ceil_divide(available_length, 2), probe_config.step_size)
    seg1_length = seg1_num_piece * probe_config.step_size
    # Select first segment

    first_piece_candidates: List[Tuple[Piece, float]] = make_first_piece_candidates(seg1_num_piece, sp_list)
    seg2_length = available_length - seg1_length
    seg2_num_seg = int(seg2_length / probe_config.step_size)
    for i1 in range(min(n1, len(first_piece_candidates))):
        first_piece, _ = first_piece_candidates[i1]
        second_piece_candidates: List[Tuple[Piece, float]] = \
            make_second_piece_candidates(seg2_num_seg, sp_list, first_piece)
        for i2 in range(min(n2, len(second_piece_candidates))):
            second_piece, _ = second_piece_candidates[i2]
            output_piece_list = rearrange_interval(first_piece, second_piece)
            yield output_piece_list


def to_tokens_and_segment_ids(query_tokens: List[str],
                              ss_list: List[ScoredPieceFromPair],
                              piece_list: List[Piece]) -> TokensAndSegmentIds:
    def get_piece_tokens(seg) -> Iterable[str]:
        return flatten(ss_list[idx].tokens for idx in range(seg.st, seg.ed))

    piece_tokens_list: Iterable[Iterable[str]] = map(get_piece_tokens, piece_list)
    doc_tokens: List[str] = list_join(piece_tokens_list, ["[SEP]"])
    tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + doc_tokens + ["[SEP]"]
    segment_ids = (len(query_tokens) + 2) * [0] + [1] * (len(doc_tokens) + 1)
    assert len(tokens) == len(segment_ids)
    return TokensAndSegmentIds(tokens, segment_ids)


def rearrange_interval(first: Piece, second: Piece) -> List[Piece]:
    if first.st <= second.st < first.ed:
        return [Piece(first.st, max(first.ed, second.ed))]
    elif second.st <= first.st < second.ed:
        return [Piece(second.st, max(first.ed, second.ed))]
    elif first.st < second.st:
        return [Piece(first.st, first.ed), Piece(second.st, second.ed)]
    elif second.st < first.st:
        return [Piece(second.st, second.ed), Piece(first.st, first.ed)]
    else:
        assert False


class IDBasedEncoder(ABC):
    @abstractmethod
    def encode(self, query_id: str, doc_id: str) -> Iterable[TokensAndSegmentIds]:
        pass


class TwoPieceSegmentComposer(IDBasedEncoder):
    def __init__(self, max_seq_length):
        self.probe_config = Config1()
        self.queries: Dict[str, str] = load_robust_04_query("desc")
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length
        qid_list = lmap(str, get_robust_qid_list())
        self.piece_score_parser = PieceScoreParser(self.queries, qid_list, self.probe_config)

    def encode(self, query_id: str, doc_id: str) -> Iterable[TokensAndSegmentIds]:
        try:
            sp_list: List[ScoredPieceFromPair] = self.piece_score_parser.get_piece_score(query_id, doc_id)
            query = self.queries[str(query_id)]
            query_tokens: List[str] = self.tokenizer.tokenize(query)
            q_term_len = len(query_tokens)
            available_length = self.max_seq_length - q_term_len - 4
            two_piece: List[Piece] = select_a_two_piece_segment(self.probe_config, available_length, sp_list)
            tas = to_tokens_and_segment_ids(query_tokens, sp_list, two_piece)
            return [tas]
        except KeyError:
            return []


class ManyTwoPieceSegmentComposer(IDBasedEncoder):
    def __init__(self, max_seq_length):
        self.probe_config = Config1()
        self.queries: Dict[str, str] = load_robust_04_query("desc")
        self.tokenizer = get_tokenizer()
        self.max_seq_length = max_seq_length
        qid_list = lmap(str, get_robust_qid_list())
        self.piece_score_parser = PieceScoreParser(self.queries, qid_list, self.probe_config)

    def encode(self, query_id: str, doc_id: str) -> Iterable[TokensAndSegmentIds]:
        try:
            sp_list: List[ScoredPieceFromPair] = self.piece_score_parser.get_piece_score(query_id, doc_id)
            query = self.queries[str(query_id)]
            query_tokens: List[str] = self.tokenizer.tokenize(query)
            q_term_len = len(query_tokens)
            available_length = self.max_seq_length - q_term_len - 4
            maybe_doc_length = self.probe_config.n_factor * self.probe_config.step_size
            n_piece = ceil_divide(maybe_doc_length, self.probe_config.max_seq_length) * 2
            n1 = n_piece
            n2 = min(n_piece - 1, 1)

            two_piece_list: Iterable[List[Piece]] = select_many_two_piece_segment(self.probe_config, available_length,
                                                                                  sp_list, n1, n2)

            def format_as_tas(two_piece):
                return to_tokens_and_segment_ids(query_tokens, sp_list, two_piece)

            return map(format_as_tas, two_piece_list)
        except KeyError:
            return []



