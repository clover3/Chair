from collections import defaultdict
from typing import List, Dict, Tuple
from typing import NamedTuple

import tlm.qtype.qe_de_res_parse
from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from list_lib import lmap
from misc_lib import group_by, average, tprint
from scipy_aux import logit_to_score_softmax
from tlm.estimator_output_reader import join_prediction_with_info


class SegmentScore(NamedTuple):
    start_idx: int
    end_idx: int
    score: float

    @classmethod
    def from_dict(cls, step_size, window_size, d):
        idx = d['idx']
        start_idx = idx * step_size
        end_idx = start_idx + window_size
        return SegmentScore(start_idx, end_idx, logit_to_score_softmax(d['logits']))

    @classmethod
    def from_dict2(cls, d):
        start_idx = d['st']
        end_idx = d['ed']
        return SegmentScore(start_idx, end_idx, logit_to_score_softmax(d['logits']))


class DocTokenScore(NamedTuple):
    query_id: str
    q_text: str
    doc_id: str
    scores: List[float]
    segment_score_list: List[SegmentScore]
    full_scores: Dict[int, List[float]]

    def max_segment_score(self):
        return max([s.score for s in self.segment_score_list])


def get_token_score(ss_list: List[SegmentScore]):
    scores_per_token: Dict[int, List] = defaultdict(list)
    reduce_fn = average

    for ss in ss_list:
        for idx in range(ss.start_idx, ss.end_idx):
            scores_per_token[idx].append(ss.score)

    max_len = max(scores_per_token.keys())
    score_arr = [0] * max_len
    for idx in range(max_len):
        scores: List[float] = scores_per_token[idx]
        score = reduce_fn(scores)
        score_arr[idx] = score
    return score_arr


def get_full_token_scores(ss_list: List[SegmentScore]) -> Dict[int, List[float]]:
    scores_per_token: Dict[int, List[float]] = defaultdict(list)

    for ss in ss_list:
        for idx in range(ss.start_idx, ss.end_idx):
            scores_per_token[idx].append(ss.score)

    return scores_per_token


def collect_token_scores(info_file_path,
                         prediction_file_path,
                         query_token_len_d,
                         step_size,
                         window_size) -> \
        List[DocTokenScore]:
    grouped_score: Dict = load_scores(info_file_path, prediction_file_path)

    out_entries: List[DocTokenScore] = []
    for pair_id, items in grouped_score.items():
        query_id, doc_id = pair_id
        query_token_len = query_token_len_d[query_id]
        content_len = window_size - query_token_len - 3
        q_text = items[0]['query'].text
        ss_list: List[SegmentScore] = list([SegmentScore.from_dict(step_size, content_len, e) for e in items])
        token_score: List[float] = get_token_score(ss_list)
        full_token_scores: Dict[int, List] = get_full_token_scores(ss_list)
        dts = DocTokenScore(query_id, q_text, doc_id, token_score, ss_list, full_token_scores)
        out_entries.append(dts)
        if len(out_entries) > 500:
            break
    return out_entries


def load_scores(info_file_path, prediction_file_path):
    input_type = "qc"
    f_handler = get_format_handler(input_type)
    tprint("Loading json info")
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    key_logit = "logits"
    tprint("Reading predictions...")
    data: List[Dict] = join_prediction_with_info(prediction_file_path, info, ["data_id", key_logit])
    grouped: Dict[Tuple[str, str], List[Dict]] = group_by(data, tlm.qtype.qe_de_res_parse.get_pair_id)
    print("number of groups:", len(grouped))
    return grouped


# max(all st) < content_len <= max(all st) + step_size

class SegmentScorePair(NamedTuple):
    long_seg: SegmentScore
    short_seg: SegmentScore
    abl_idx_st: int
    abl_idx_ed: int

    def grouping_key(self):
        return self.abl_idx_st, self.abl_idx_ed

    @classmethod
    def init_left_ablation(cls, long_seg: SegmentScore, short_seg: SegmentScore):
        abl_idx_st = long_seg.start_idx
        abl_idx_ed = short_seg.start_idx
        assert long_seg.end_idx == short_seg.end_idx
        return SegmentScorePair(long_seg, short_seg, abl_idx_st, abl_idx_ed)

    @classmethod
    def init_right_ablation(cls, long_seg: SegmentScore, short_seg: SegmentScore):
        abl_idx_st = short_seg.end_idx
        abl_idx_ed = long_seg.end_idx
        assert long_seg.start_idx == short_seg.start_idx
        return SegmentScorePair(long_seg, short_seg, abl_idx_st, abl_idx_ed)

    def is_left_ablation(self):
        return self.long_seg.end_idx == self.short_seg.end_idx

    def is_right_ablation(self):
        return self.long_seg.start_idx == self.short_seg.start_idx

    def get_score_diff(self) -> float:
        return self.long_seg.score - self.short_seg.score

    def get_max_score(self):
        return max(self.long_seg.score, self.short_seg.score)


def pair_ss(ss_list: List[SegmentScore], step_size) -> List[SegmentScorePair]:
    all_ss_pairs: List[SegmentScorePair] = []
    for e1 in ss_list:
        for e2 in ss_list:
            if e1.start_idx + step_size == e2.start_idx and e1.end_idx == e2.end_idx:
                sspair = SegmentScorePair.init_left_ablation(e1, e2)
                all_ss_pairs.append(sspair)
            if e1.start_idx == e2.start_idx and e1.end_idx == e2.end_idx + step_size:
                sspair = SegmentScorePair.init_right_ablation(e1, e2)
                all_ss_pairs.append(sspair)
    return all_ss_pairs


def cap_ed(ss_list: List[SegmentScore], step_size) -> List[SegmentScore]:
    max_start_idx = max([s.start_idx for s in ss_list])
    cap_end_idx = max_start_idx + step_size

    def transform(ss: SegmentScore):
        if ss.end_idx < cap_end_idx:
            return ss
        else:
            return SegmentScore(ss.start_idx, cap_end_idx, ss.score)

    return lmap(transform, ss_list)


def get_token_score_from_sspairs(ss_pairs: List[SegmentScorePair]) -> Dict[int, List[SegmentScorePair]]:
    grouped: Dict[Tuple[int, int], List[SegmentScorePair]] = group_by(ss_pairs, SegmentScorePair.grouping_key)
    token_to_pair_d = defaultdict(list)
    for key, ss_pairs in grouped.items():
        for ss_pair in ss_pairs:
            for idx in range(ss_pair.abl_idx_st, ss_pair.abl_idx_ed):
                token_to_pair_d[idx].append(ss_pair)
    return token_to_pair_d


class AnalyzedDoc(NamedTuple):
    query_id: str
    doc_id: str
    token_info: Dict[int, List[SegmentScorePair]]


def token_score_by_ablation(info_file_path,
                            prediction_file_path,
                            query_token_len_d,
                            step_size,
                            window_size) -> List[AnalyzedDoc]:
    grouped_score: Dict = load_scores(info_file_path, prediction_file_path)

    out_entries: List[AnalyzedDoc] = []
    for pair_id, items in grouped_score.items():
        ss_list: List[SegmentScore] = lmap(SegmentScore.from_dict2, items)
        ss_list = cap_ed(ss_list, step_size)
        query_id, doc_id = pair_id
        ss_pairs: List[SegmentScorePair] = pair_ss(ss_list, step_size)
        token_score: Dict[int, List[SegmentScorePair]] = get_token_score_from_sspairs(ss_pairs)
        ad = AnalyzedDoc(query_id, doc_id, token_score)
        out_entries.append(ad)
        if len(out_entries) > 500:
            break
    return out_entries
