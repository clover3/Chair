import os
from typing import List, Dict

import tensorflow as tf
from omegaconf import OmegaConf

from adhoc.bm25_class import BM25Bare
from adhoc.other.bm25_retriever_helper import get_tokenize_fn, get_bm25_stats_from_conf
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import left
from misc_lib import get_dir_files, batch_iter_from_entry_iter, path_join, get_second
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import create_dataset_common
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.per_project.transparency.misc_common import load_table
from trainer_v2.per_project.transparency.mmp.pep_to_tt.bm25_match_analyzer import BM25_MatchAnalyzer
from trainer_v2.per_project.transparency.mmp.pep_to_tt.bm25_match_analyzer2 import BM25_MatchAnalyzer2
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_common import get_pep_predictor
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_encoders import PEP_TT_EncoderSingle, PEP_TT_EncoderIF, \
    PEP_TT_Encoder2
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


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


def read_pep_tt_dataset(
        file_path,
        run_config: RunConfig2,
        seq_len,
        is_for_training,
    ) -> tf.data.Dataset:
    int_list_items = ["pos_input_ids", "pos_segment_ids", "neg_input_ids", "neg_segment_ids"]
    float_items = ["pos_multiplier", "pos_value_score", "pos_norm_add_factor",
                   "neg_multiplier", "neg_value_score", "neg_norm_add_factor"]

    return read_dataset_int_list_float_items(
        file_path, run_config, seq_len, is_for_training, int_list_items, float_items)


def read_pep_tt_dataset2(
        file_path,
        run_config: RunConfig2,
        seq_len,
        is_for_training,
    ) -> tf.data.Dataset:
    int_list_items = ["pos_input_ids", "pos_segment_ids", "neg_input_ids", "neg_segment_ids"]
    float_items = ["pos_multiplier", "pos_value_score", "pos_norm_add_factor", "pos_tf",
                   "neg_multiplier", "neg_value_score", "neg_norm_add_factor", "neg_tf"]

    return read_dataset_int_list_float_items(
        file_path, run_config, seq_len, is_for_training, int_list_items, float_items)


def read_dataset_int_list_float_items(
        file_path, run_config, seq_len, is_for_training,
        int_list_items, float_items):

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


def get_pep_tt_single_encoder(model_config: PEP_TT_ModelConfig, conf):
    bert_tokenizer = get_tokenizer()
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    avdl, cdf, df, dl_d = get_bm25_stats_from_conf(bm25_conf, None)
    get_pep_top_k = get_pep_predictor(conf)

    bm25 = BM25Bare(df, len(dl_d), avdl, b=1.4)
    bm25_analyzer = BM25_MatchAnalyzer(bm25, get_pep_top_k, bert_tokenizer.basic_tokenizer.tokenize)

    term_to_subword = bert_tokenizer.tokenize
    return PEP_TT_EncoderSingle(bert_tokenizer, model_config, bm25_analyzer, term_to_subword)


def get_pep_tt_single_encoder_for_with_align_info(
        model_config: PEP_TT_ModelConfig,
        conf):
    bert_tokenizer = get_tokenizer()
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    avdl, cdf, df, dl_d = get_bm25_stats_from_conf(bm25_conf, None)
    get_pep_top_k = None
    bm25 = BM25Bare(df, len(dl_d), avdl, bm25_conf.k1, bm25_conf.k2, bm25_conf.b)
    bm25_tokenizer = get_tokenize_fn(bm25_conf)
    bm25_analyzer = BM25_MatchAnalyzer(bm25, get_pep_top_k, bm25_tokenizer)
    term_to_subword = bert_tokenizer.tokenize
    return PEP_TT_EncoderSingle(bert_tokenizer, model_config, bm25_analyzer, term_to_subword)


def get_pep_tt_single_encoder2(
        model_config: PEP_TT_ModelConfig,
        conf):
    bert_tokenizer = get_tokenizer()
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    avdl, cdf, df, dl_d = get_bm25_stats_from_conf(bm25_conf, None)
    get_pep_top_k = None
    bm25 = BM25Bare(df, len(dl_d), avdl, bm25_conf.k1, bm25_conf.k2, bm25_conf.b)
    bm25_tokenizer = get_tokenize_fn(bm25_conf)
    bm25_analyzer = BM25_MatchAnalyzer2(bm25, get_pep_top_k, bm25_tokenizer)
    term_to_subword = bert_tokenizer.tokenize
    return PEP_TT_Encoder2(bert_tokenizer, model_config, bm25_analyzer, term_to_subword)


class PEP_TT_DatasetBuilderWithAlignInfo:
    def __init__(
            self,
            encoder: PEP_TT_EncoderSingle,
            conf,
            batch_size):
        self.batch_size = batch_size
        self.encoder = encoder
        self.conf = conf
        self.align_info_conf = OmegaConf.load(conf.align_info_conf)
        self.output_d: Dict[str, Dict[str, float]] = {}

    def get_pep_top_k(self, q_term, d_term_iter) -> List[str]:
        score_d_per_q_term = self.output_d[q_term]
        d_term_list = list(d_term_iter)
        scores = []
        for d_term in d_term_list:
            try:
                scores.append((d_term, score_d_per_q_term[d_term]))
            except KeyError:
                pass

        scores.sort(key=get_second, reverse=True)
        return left(scores)

    def get_pep_tt_dataset(
            self,
            _file_path,
            is_training,
        ) -> tf.data.Dataset:
        c_log.info("File path is ignored")

        def generator():
            raw_train_iter = tsv_iter(self.align_info_conf.qd_triplet_file)
            partition_size = self.align_info_conf.line_per_job
            batch_itr = batch_iter_from_entry_iter(raw_train_iter, partition_size)
            for idx, partition in enumerate(batch_itr):
                table_path = path_join(self.align_info_conf.score_save_dir, f"{idx}.txt")
                if not os.path.exists(table_path) or os.path.getsize(table_path) == 0:
                    c_log.warn("File does not exists skip this parition %s", table_path)
                    continue

                self.output_d = load_table(table_path)
                self.encoder.bm25_analyzer.get_pep_top_k = self.get_pep_top_k
                for row in partition:
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
