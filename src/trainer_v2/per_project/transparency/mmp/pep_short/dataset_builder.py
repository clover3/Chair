import collections
from dataclasses import dataclass

import tensorflow as tf

from data_generator.create_feature import create_int_feature, create_float_feature
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad
from misc_lib import get_dir_files, get_second, pick1
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfigType

# Roles:
# Encoder is responsible for convert text/numbers into BERT format
# Dataset is builder for reading file and extracting/enumerating text/numbers



class PepShortEncoder:
    def __init__(self,
                 bert_tokenizer,
                 model_config: ModelConfigType,
                 ):
        self.model_config = model_config
        self.tokenizer = bert_tokenizer

    def encode(self, qt: str, dt1: str, dt2: str, s1, s2):
        input_ids1, segment_ids1 = combine_with_sep_cls_and_pad(
            self.tokenizer,
            self.tokenizer.tokenize(qt),
            self.tokenizer.tokenize(dt1),
            self.model_config.max_seq_length)

        input_ids2, segment_ids2 = combine_with_sep_cls_and_pad(
            self.tokenizer,
            self.tokenizer.tokenize(qt),
            self.tokenizer.tokenize(dt2),
            self.model_config.max_seq_length)

        return {
            'input_ids1': input_ids1,
            'segment_ids1': segment_ids1,
            'input_ids2': input_ids2,
            'segment_ids2': segment_ids2,
            's1': s1,
            's2': s2,
        }

    def get_output_signature(self):
        max_seq_len = self.model_config.max_seq_length
        ids_spec = tf.TensorSpec(shape=(max_seq_len,), dtype=tf.int64)
        output_signature_per_qd = {
            'input_ids1': ids_spec,
            'segment_ids1': ids_spec,
            's1': tf.TensorSpec(shape=(), dtype=tf.float32),
            'input_ids2': ids_spec,
            'segment_ids2': ids_spec,
            's2': tf.TensorSpec(shape=(), dtype=tf.float32),
        }

        return output_signature_per_qd

    def to_tf_feature(self, feature) -> collections.OrderedDict:
        features = collections.OrderedDict()
        for k, v in feature.items():
            if type(v) == list:
                features[k] = create_int_feature(v)
            elif type(v) == float:
                features[k] = create_float_feature([v])
            else:
                raise ValueError()
        return features


class PEPShortDatasetBuilder:
    def __init__(self, encoder: PepShortEncoder, batch_size):
        self.batch_size = batch_size
        self.encoder = encoder

    def get_pep_tt_dataset(
            self,
            dir_path,
            is_training,
    ) -> tf.data.Dataset:
        c_log.debug("get_pep_tt_dataset")
        file_list = get_dir_files(dir_path)

        def generator():
            for file_path in file_list:
                raw_train_iter = tsv_iter(file_path)
                c_log.debug("Open %s", file_path)
                for idx, row in enumerate(raw_train_iter):
                    qt = row[0]
                    i = 1
                    dt_score_list = []
                    while i < len(row):
                        dt = row[i]
                        score = float(row[i + 1])
                        dt_score_list.append((dt, score))
                        i += 2

                    dt_score_list.sort(key=get_second, reverse=True)
                    dt1, s1 = dt_score_list[0]
                    dt2, s2 = pick1(dt_score_list[1:])
                    d = self.encoder.encode(qt, dt1, dt2, s1, s2)
                    yield d

        output_signature = self.encoder.get_output_signature()
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.batch(self.batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
