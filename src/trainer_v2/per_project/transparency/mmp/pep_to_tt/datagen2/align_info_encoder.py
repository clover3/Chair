import collections
import json
import math
from dataclasses import dataclass
from typing import Optional, OrderedDict, Iterator, Iterable

import tensorflow as tf
from omegaconf import OmegaConf

from adhoc.bm25_class import BM25Bare
from data_generator.create_feature import create_int_feature, create_float_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TextRep
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_encoders import TTSingleTrainFeature, DocScoringSingle
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


class AlignInfo:
    def __init__(self, d):
        self.data_id = d['data_id']
        self.q_term = d['q_term']
        self.cands = d['cands']

    def get_key(self):
        return self.data_id


@dataclass
class TripletAlign:
    q: str
    d_pos: str
    d_neg: str
    pos_align: Optional[AlignInfo]
    neg_align: Optional[AlignInfo]


class BM25_MatchAnalyzerWAlignInfo:
    def __init__(self, conf):
        bm25_conf = OmegaConf.load(conf.bm25conf_path)
        avdl, cdf, df, dl_d = get_bm25_stats_from_conf(bm25_conf, None)
        self.bm25 = BM25Bare(df, len(dl_d), avdl, b=1.4)
        self.tokenizer = get_tokenizer()

    def apply(self, q, d, align_info):
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
            if exact_match_cnt:
                score_per_q_term: float = bm25.per_term_score(
                    q_term, qtf, exact_match_cnt, d.get_sp_size())
                value_score += score_per_q_term
            elif align_info is not None:
                if align_info.q_term == q_term and align_info.cands:
                    # We use top score only
                    multiplier = query_factor(q_term, qtf) * denom_factor
                    d_term, _score = align_info.cands[0]
                    per_term_entry = {
                        'q_term': q_sp_sb_mapping[q_term],
                        'd_term': d_sp_sb_mapping[d_term],
                        'multiplier': multiplier,
                        'q_term_raw': q_term
                    }
                    per_unknown_tf.append(per_term_entry)
        return K, per_unknown_tf, value_score


class PEP_TT_Encoder7:
    def __init__(self, model_config: PEP_TT_ModelConfig, conf):
        self.bm25_analyzer = BM25_MatchAnalyzerWAlignInfo(conf)
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

    def _get_doc_score_factors(
            self, q: str, d: str, align_info: Optional[AlignInfo]) -> DocScoringSingle:
        K, per_unknown_tf, value_score = self.bm25_analyzer.apply(q, d, align_info)

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

    def _encode_triplet(self, item: TripletAlign) -> TTSingleTrainFeature:
        s_pos = self._get_doc_score_factors(item.q, item.d_pos, item.pos_align)
        s_neg = self._get_doc_score_factors(item.q, item.d_neg, item.neg_align)
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

    def _to_tf_feature(self, feature: TTSingleTrainFeature) -> collections.OrderedDict:
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

    def encode(self, item: TripletAlign) -> OrderedDict:
        d = self._encode_triplet(item)
        return self._to_tf_feature(d)


def json_iterator(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as JSON
            try:
                data = json.loads(line)
                yield data
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}\nError: {e}")


def parse_align_info(item, decoding_dict):
    # Decode 'q_term_id'
    q_term_id = item['q_term_id']
    decoded_q_term = decoding_dict[q_term_id]

    # Decode each candidate in 'cands'
    decoded_cands = []
    for cand in item['cands']:
        decoded_cand = (decoding_dict[cand[0]], cand[1])
        decoded_cands.append(decoded_cand)

    # Construct the output
    decoded_item = {
        'data_id': item['data_id'],
        'q_term': decoded_q_term,
        'cands': decoded_cands
    }
    return AlignInfo(decoded_item)


def join_triplet_with_align(
        align_info: Iterator[AlignInfo],
        qdd_iter: Iterable[tuple[str, str, str]]) -> Iterator[TripletAlign]:
    try:
        last_item: AlignInfo = next(align_info)
        for idx, triplet in enumerate(qdd_iter):
            data_id_pos = idx * 10 + 1
            data_id_neg = idx * 10 + 2

            def get_align_item_or_move(data_id) -> Optional[AlignInfo]:
                nonlocal last_item
                key = last_item.get_key()
                if key == data_id:
                    align_item = last_item
                    last_item = next(align_info)
                else:
                    assert data_id < key
                    align_item = None
                return align_item

            pos_align = get_align_item_or_move(data_id_pos)
            neg_align = get_align_item_or_move(data_id_neg)
            q, d_pos, d_neg = triplet
            yield TripletAlign(q, d_pos, d_neg, pos_align, neg_align)
    except StopIteration:
        pass
