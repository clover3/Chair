import collections
import dataclasses
import math
from abc import ABC, abstractmethod
from typing import TypedDict

import tensorflow as tf
from omegaconf import OmegaConf

from adhoc.bm25_class import BM25Bare
from data_generator.create_feature import create_int_feature, create_float_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf
from misc_lib import path_join, get_dir_files
from table_lib import tsv_iter
from tlm.data_gen.base import concat_tuple_windows
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import create_dataset_common
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TextRep
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_common import get_pep_predictor
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


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


class PEP_TT_EncoderIF(ABC):
    @abstractmethod
    def encode_triplet(self, q: str, d_pos: str, d_neg: str) -> dict:
        pass

    @abstractmethod
    def get_output_signature(self):
        pass


@dataclasses.dataclass
class DocScoring:
    value_score: float
    q_term_arr: list[list[str]]
    d_term_arr: list[list[str]]
    multiplier_arr: list[float]
    norm_add_factor: float


@dataclasses.dataclass
class DocScoringSingle:
    value_score: float  # Scores from exact match
    q_term: list[str]   # query term that we want to train
    d_term: list[str]   # document term that we want to train
    multiplier: float   # factors of BM25 that are multiplied to term frequency to  get score
    norm_add_factor: float  # K-value, which works for the document length penalty.


class PEP_TT_EncoderSingle(PEP_TT_EncoderIF):
    def __init__(self, model_config: PEP_TT_ModelConfig, conf):
        self.bm25_analyzer = BM25_MatchAnalyzer(conf)
        self.bm25 = self.bm25_analyzer.bm25
        self.model_config = model_config
        self.tokenizer = get_tokenizer()

    def _encode_one(self, s: DocScoringSingle) -> dict[str, list]:
        max_seq_len = self.model_config.max_seq_length
        q_term = s.q_term
        d_term = s.d_term
        input_ids, segment_ids = combine_with_sep_cls_and_pad(
            self.tokenizer, q_term, d_term, max_seq_len)
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

        def get_df(entry):
            return self.bm25.df[entry["q_term_raw"]]

        per_unknown_tf.sort(key=get_df)
        if per_unknown_tf:
            item = per_unknown_tf[0]
            output = DocScoringSingle(
                value_score=value_score,
                q_term=item['q_term'],
                d_term=item['d_term'],
                multiplier=item['multiplier'],
                norm_add_factor=K
            )
        else:
            output = DocScoringSingle(
                value_score=value_score,
                q_term=[],
                d_term=[],
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
    def __init__(self, model_config: PEP_TT_ModelConfig, conf):
        self.bm25_analyzer = BM25_MatchAnalyzer(conf)
        self.model_config = model_config
        self.tokenizer = get_tokenizer()

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
            q_term_arr=[item['q_term'] for item in per_unknown_tf],
            d_term_arr=[item['d_term'] for item in per_unknown_tf],
            multiplier_arr=[item['multiplier'] for item in per_unknown_tf],
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


class PEP_TT_DatasetBuilder:
    def __init__(self, encoder: PEP_TT_EncoderIF, batch_size):
        self.batch_size = batch_size
        self.encoder = encoder

    def get_pep_tt_dataset(
            self,
            dir_path,
            is_training,
        ) -> tf.data.Dataset:
        file_list = get_dir_files(dir_path)

        def generator():
            for file_path in file_list:
                raw_train_iter = tsv_iter(file_path)
                for row in raw_train_iter:
                    q = row[0]
                    d_pos = row[1]
                    d_neg = row[2]
                    feature_d = self.encoder.encode_triplet(q, d_pos, d_neg)
                    yield feature_d

        output_signature = self.encoder.get_output_signature()
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.batch(self.batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


class BM25_MatchAnalyzer:
    def __init__(self, conf):
        self.get_pep_top_k = get_pep_predictor(conf)
        bm25_conf = OmegaConf.load(conf.bm25conf_path)
        avdl, cdf, df, dl_d = get_bm25_stats_from_conf(bm25_conf, None)
        self.bm25 = BM25Bare(df, len(dl_d), avdl, b=1.4)
        self.tokenizer = get_tokenizer()

    def apply(self, q, d):
        tokenizer = self.tokenizer
        bm25 = self.bm25
        q = TextRep.from_text(tokenizer, q)
        d: TextRep = TextRep.from_text(tokenizer, d)
        q_sp_sb_mapping = q.tokenized_text.get_sp_to_sb_map()
        d_sp_sb_mapping = d.tokenized_text.get_sp_to_sb_map()

        def query_factor(q_term, qf) -> float:
            N = bm25.N
            df = bm25.df[q_term]
            idf_like = math.log((N - df + 0.5) / (df + 0.5) + 1)
            qft_based = ((bm25.k2 + 1) * qf) / (bm25.k2 + qf)
            return idf_like * qft_based

        dl = d.get_sp_size()
        denom_factor = (1 + bm25.k1)
        K = bm25.k1 * ((1 - bm25.b) + bm25.b * (float(dl) / float(bm25.avdl)))
        per_unknown_tf: list[dict] = []
        value_score = 0.0
        for q_term, qtf, _ in q.get_bow():
            exact_match_cnt: int = d.counter[q_term]
            top_k_terms: list[str] = self.get_pep_top_k(q_term, d.counter.keys())

            if exact_match_cnt:
                score_per_q_term: float = bm25.per_term_score(
                    q_term, qtf, exact_match_cnt, d.get_sp_size())
                value_score += score_per_q_term
            elif top_k_terms:
                top_term = top_k_terms[0]
                # We use top score only
                multiplier = query_factor(q_term, qtf) * denom_factor
                per_term_entry = {
                    'q_term': q_sp_sb_mapping[q_term],
                    'd_term': d_sp_sb_mapping[top_term],
                    'multiplier': multiplier,
                    'q_term_raw': q_term
                }
                per_unknown_tf.append(per_term_entry)
        return K, per_unknown_tf, value_score


def read_pep_tt_dataset(
        file_path,
        run_config: RunConfig2,
        seq_len,
        is_for_training,
    ) -> tf.data.Dataset:
    int_list_items = ["pos_input_ids", "pos_segment_ids", "neg_input_ids", "neg_segment_ids"]
    float_items = ["pos_multiplier", "pos_value_score", "pos_norm_add_factor",
                   "neg_multiplier", "neg_value_score", "neg_norm_add_factor"]

    def decode_record(record):
        name_to_features = {}
        for key in int_list_items:
            name_to_features[key] = tf.io.FixedLenFeature([seq_len], tf.int64)
        for key in float_items:
            name_to_features[key] = tf.io.FixedLenFeature([1], tf.float32)
        record = tf.io.parse_single_example(record, name_to_features)
        return record

    dataset = create_dataset_common(
        decode_record,
        run_config,
        file_path,
        is_for_training)
    return dataset

