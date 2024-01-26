import collections
import dataclasses
import itertools
import random
from abc import ABC, abstractmethod
from typing import TypedDict

import tensorflow as tf

from data_generator.create_feature import create_int_feature, create_float_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad
from misc_lib import pick1
from table_lib import tsv_iter
from tlm.data_gen.base import concat_tuple_windows
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.bm25_match_analyzer import TermAlignInfo
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig

from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

class TTSingleTrainFeature(TypedDict):
    pos_input_ids: list[int]
    pos_segment_ids: list[int]
    pos_multiplier: float
    pos_value_score: float
    pos_norm_add_factor: float
    neg_input_ids: list[int]
    neg_segment_ids: list[int]
    neg_multiplier: float
    neg_value_score: float
    neg_norm_add_factor: float


@dataclasses.dataclass
class DocScoring:
    value_score: float
    q_term_arr: list[str]
    d_term_arr: list[str]
    multiplier_arr: list[float]
    norm_add_factor: float


Subword = str


@dataclasses.dataclass
class DocScoringSingle:
    value_score: float  # Scores from exact match
    q_term: str   # query term that we want to train
    d_term: str   # document term that we want to train
    multiplier: float   # factors of BM25 that are multiplied to term frequency to  get score
    norm_add_factor: float  # K-value, which works for the document length penalty.


class PEP_TT_EncoderIF(ABC):
    @abstractmethod
    def encode_triplet(self, q: str, d_pos: str, d_neg: str) -> dict:
        pass

    @abstractmethod
    def get_output_signature(self):
        pass


class PEP_TT_EncoderSingle(PEP_TT_EncoderIF):
    def __init__(self,
                 bert_tokenizer,
                 model_config: PEP_TT_ModelConfig,
                 bm25_analyzer,
                 term_to_subwords
                 ):
        self.model_config = model_config
        self.tokenizer = bert_tokenizer
        self.bm25_analyzer = bm25_analyzer
        self.term_to_subwords = term_to_subwords
    def _encode_one(self, s: DocScoringSingle) -> dict[str, list]:
        max_seq_len = self.model_config.max_seq_length
        q_term_sb_rep = self.term_to_subwords(s.q_term)
        d_term_sb_rep = self.term_to_subwords(s.d_term)
        input_ids, segment_ids = combine_with_sep_cls_and_pad(
            self.tokenizer, q_term_sb_rep, d_term_sb_rep, max_seq_len)
        return {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "multiplier": s.multiplier,
            "value_score": s.value_score,
            "norm_add_factor": s.norm_add_factor,
        }

    def _encode_pair(self, s_pos, s_neg) -> dict[str, list]:
        doc_d = {
            "pos": s_pos,
            "neg": s_neg,
        }
        all_features = {}
        for role, doc in doc_d.items():  # For pos neg
            for key, value in self._encode_one(doc).items():
                all_features[f"{role}_{key}"] = value

        return all_features

    def _get_doc_score_factors(self, q: str, d: str) -> DocScoringSingle:
        K, per_unknown_tf, value_score = self.bm25_analyzer.apply(q, d)

        random.shuffle(per_unknown_tf)
        if per_unknown_tf:
            item: TermAlignInfo = per_unknown_tf[0]
            output = DocScoringSingle(
                value_score=value_score,
                q_term=item.q_term,
                d_term=item.d_term,
                multiplier=item.multiplier,
                norm_add_factor=K
            )
        else:
            output = DocScoringSingle(
                value_score=value_score,
                q_term="",
                d_term="",
                multiplier=0.,
                norm_add_factor=K
            )
        return output

    def encode_triplet(self, q: str, d_pos: str, d_neg: str) -> TTSingleTrainFeature:
        s_pos = self._get_doc_score_factors(q, d_pos)
        s_neg = self._get_doc_score_factors(q, d_neg)
        feature_d: TTSingleTrainFeature = self._encode_pair(s_pos, s_neg)
        return feature_d

    def get_output_signature(self):
        max_seq_len = self.model_config.max_seq_length
        ids_spec = tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int32)
        output_signature_per_qd = {
            'input_ids': ids_spec,
            'segment_ids': ids_spec,
            'multiplier': tf.TensorSpec(shape=(), dtype=tf.float32),
            'value_score': tf.TensorSpec(shape=(), dtype=tf.float32),
            'norm_add_factor': tf.TensorSpec(shape=(), dtype=tf.float32),
        }

        output_signature = {}
        for role in ["pos", "neg"]:
            for key, value in output_signature_per_qd.items():
                output_signature[f"{role}_{key}"] = value
        return output_signature

    def to_tf_feature(self, feature: TTSingleTrainFeature) -> collections.OrderedDict:
        # Feature values are either int list or float value
        features = collections.OrderedDict()
        for k, v in feature.items():
            if type(v) == list:
                features[k] = create_int_feature(v)
            elif type(v) == float:
                features[k] = create_float_feature([v])
            else:
                raise ValueError()
        return features


class PEP_TT_EncoderMulti(PEP_TT_EncoderIF):
    def __init__(self,
                 bert_tokenizer,
                 model_config: PEP_TT_ModelConfig,
                 bm25_analyzer):
        self.tokenizer = get_tokenizer()
        self.model_config = model_config
        self.tokenizer = bert_tokenizer
        self.bm25_analyzer = bm25_analyzer

    def _encode_one(self, s: DocScoring) -> dict[str, list]:
        max_term_pair = self.model_config.max_num_terms
        max_seq_len = self.model_config.max_seq_length
        tuple_list = []
        input_ids_all = []
        segment_ids_all = []
        for i in range(max_term_pair):
            try:
                q_term = s.q_term_arr[i]
                d_term = s.d_term_arr[i]
                if len(q_term) + len(d_term) + 1 > max_seq_len:
                    c_log.warn("Long sequence of length %d", len(q_term) + len(d_term) + 1)
                    pass
            except IndexError:
                q_term = []
                d_term = []
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                self.tokenizer, q_term, d_term, max_seq_len)
            tuple_list.append((input_ids, segment_ids))
            input_ids_all.append(input_ids)
            segment_ids_all.append(segment_ids)

        input_ids, segment_ids = concat_tuple_windows(tuple_list, max_seq_len)
        multiplier_arr = s.multiplier_arr[:max_term_pair]
        pad_len = max_term_pair - len(multiplier_arr)
        multiplier_arr = multiplier_arr + [0] * pad_len
        return {
            "input_ids": input_ids_all,
            "segment_ids": segment_ids_all,
            "multiplier_arr": multiplier_arr,
            "value_score": s.value_score,
            "norm_add_factor": s.norm_add_factor,
        }

    def _encode_pair(self, s_pos, s_neg) -> dict[str, list]:
        doc_d = {
            "pos": s_pos,
            "neg": s_neg,
        }
        all_features = {}
        for role, doc in doc_d.items():  # For pos neg
            for key, value in self._encode_one(doc).items():
                all_features[f"{role}_{key}"] = value

        return all_features

    def _get_doc_score_factors(self, q: str, d: str) -> DocScoring:
        K, per_unknown_tf, value_score = self.bm25_analyzer.apply(q, d)
        output = DocScoring(
            value_score=value_score,
            q_term_arr=[item.q_term for item in per_unknown_tf],
            d_term_arr=[item.d_term for item in per_unknown_tf],
            multiplier_arr=[item.multiplier for item in per_unknown_tf],
            norm_add_factor=K
        )
        return output

    def encode_triplet(self, q: str, d_pos: str, d_neg: str) -> TTSingleTrainFeature:
        s_pos = self._get_doc_score_factors(q, d_pos)
        s_neg = self._get_doc_score_factors(q, d_neg)
        feature_d: TTSingleTrainFeature = self._encode_pair(s_pos, s_neg)
        return feature_d

    def get_output_signature(self):
        max_term_pair = self.model_config.max_num_terms
        max_seq_len = self.model_config.max_seq_length
        int_2d_list_spec = tf.TensorSpec(shape=(max_term_pair, max_seq_len,), dtype=tf.int32)
        output_signature_per_qd = {
            'input_ids': int_2d_list_spec,
            'segment_ids': int_2d_list_spec,
            'multiplier_arr': tf.TensorSpec(shape=(max_term_pair,), dtype=tf.float32),
            'value_score': tf.TensorSpec(shape=(), dtype=tf.float32),
            'norm_add_factor': tf.TensorSpec(shape=(), dtype=tf.float32),
        }

        output_signature = {}
        for role in ["pos", "neg"]:
            for key, value in output_signature_per_qd.items():
                output_signature[f"{role}_{key}"] = value
        return output_signature


class AlignCandidateExtractor:
    def __init__(self, tokenize_fn):
        self.tokenize_fn = tokenize_fn

    def apply(self, q: str, d: str) -> set[tuple[str, str]]:
        q_tokens: list[str] = self.tokenize_fn(q)
        d_tokens: list[str] = self.tokenize_fn(d)

        q_counter = collections.Counter(q_tokens)
        d_counter = collections.Counter(d_tokens)

        d_terms = list(d_counter.keys())

        unique_align_check_list = set()
        not_matching_q_term = []
        for q_term, qtf in q_counter.items():
            exact_match_cnt: int = d_counter[q_term]
            if not exact_match_cnt:
                not_matching_q_term.append(q_term)

        if not_matching_q_term:
            random.shuffle(not_matching_q_term)
            q_term = not_matching_q_term[0]
            for d_term in d_terms:
                unique_align_check_list.add((q_term, d_term))
        return unique_align_check_list

    def pre_analyze_print_unique(self, itr, save_path):
        unique_align_check_list = set()
        for idx, row in enumerate(itr):
            q = row[0]
            d_pos = row[1]
            d_neg = row[2]
            for d in [d_pos, d_neg]:
                ret = self.apply(q, d)
                print(ret)
                unique_align_check_list.update(ret)

            if idx % 1000 == 0:
                print("Line={} / {} items".format(idx, len(unique_align_check_list)))


        with open(save_path, "w") as f:
            for qt, dt in unique_align_check_list:
                f.write(f"{qt}\t{dt}\n")

    def apply_all(self, q: str, d: str) -> Iterable[list[str]]:
        q_tokens: list[str] = self.tokenize_fn(q)
        d_tokens: list[str] = self.tokenize_fn(d)

        q_counter = collections.Counter(q_tokens)
        d_counter = collections.Counter(d_tokens)

        d_terms: list[str] = list(d_counter.keys())

        not_matching_q_term = []
        for q_term, qtf in q_counter.items():
            exact_match_cnt: int = d_counter[q_term]
            if not exact_match_cnt:
                not_matching_q_term.append(q_term)

        if not_matching_q_term:
            q_term = pick1(not_matching_q_term)
            yield [q_term] + d_terms
        else:
            yield []

    def pre_analyze_print_per_query(self, itr, save_path):
        with open(save_path, "w") as f:
            for idx, row in enumerate(itr):
                q = row[0]
                d_pos = row[1]
                d_neg = row[2]
                for d in [d_pos, d_neg]:
                    for qt_dts in self.apply_all(q, d):
                        f.write("\t".join(qt_dts) + "\n")
                f.flush()