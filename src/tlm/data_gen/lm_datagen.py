import collections
import os
import pickle
import random
import time
from functools import partial

import tensorflow as tf

from cpath import data_path
from data_generator import tokenizer_wo_tf as tokenization
from list_lib import flatten
from misc_lib import pick1
from tf_util.record_writer_wrap import RecordWriterWrap
from tf_util.tf_logging import tf_logging
from tlm.data_gen import bert_data_gen as btd
from tlm.data_gen.base import get_basic_input_feature, format_tokens_n_segid, \
    truncate_seq_pair, format_tokens_pair_n_segid, truncate_seq
from tlm.data_gen.tf_logger_misc import log_print_inst


class OrderedDictBuilder(collections.OrderedDict):
    def extend(self, other_dict):
        for key, value in other_dict.items():
            self.update({key: value})


class MLMFeaturizer:
    def __init__(self, tokenizer, max_seq_length, max_predictions_per_seq):
        self.get_basic_input_features = partial(get_basic_input_feature, tokenizer, max_seq_length)
        self.get_masked_lm_features = partial(get_masked_lm_features, tokenizer, max_predictions_per_seq)

    def instance_to_features(self, instance):
        basic_features = self.get_basic_input_features(instance.tokens, instance.segment_ids)
        lm_mask_features = self.get_masked_lm_features(instance.masked_lm_positions, instance.masked_lm_labels)
        features = OrderedDictBuilder()
        features.extend(basic_features)
        features.extend(lm_mask_features)
        next_sentence_label = 1 if instance.is_random_next else 0
        features["next_sentence_labels"] = btd.create_int_feature([next_sentence_label])
        return features


class LMTrainBase:
    def __init__(self):
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.masked_lm_prob = 0.15
        self.max_seq_length = 512
        self.dupe_factor = 1
        self.rng = random.Random(time.time())

    def pool_tokens(self, document, target_seq_length, skip=False):
        return pool_tokens(self.max_seq_length, self.rng, document, target_seq_length, skip)

    def pool_chunks(self, document, target_seq_length, skip = False):
        results = []
        current_chunk = []
        current_length = 0
        i = 0
        if skip:
            i = i + self.rng.randint(0,3)
        while i < len(document):
            segment = document[i]
            if len(segment) > self.max_seq_length:
                segment = segment[:self.max_seq_length]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                results.append(current_chunk)
                current_chunk = []
                current_length = 0
                if skip:
                    i = i + self.rng.randint(0, 3)
            i += 1
        return results

    def pool_chunks_from_docs(self, documents, target_seq_length):
        target_inst_num = 0
        docs_as_chunks = []
        for doc in documents:
            chunks = self.pool_chunks(doc, target_seq_length)
            docs_as_chunks.append(chunks)
            target_inst_num += len(chunks)
        docs_as_chunks = list([d for d in docs_as_chunks if d])
        return docs_as_chunks, target_inst_num



class LMTrainGen(LMTrainBase):
    def __init__(self):
        super(LMTrainGen, self).__init__()
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

        self.short_seq_prob = 0.1
        self.problem_per_job = 100 * 1000
        self.max_predictions_per_seq = int(self.max_seq_length * self.masked_lm_prob)

    def load_subset_documents(self, start, end):
        all_docs = []
        for i in range(start, end):
            l = self.load_doc_seg(i)
            all_docs.extend(l)
        return all_docs

    def load_doc_seg(self, doc_id):
        file_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/tokens/enwiki_train_tokens.{}".format(doc_id)
        f = open(file_path.format(doc_id), "rb")
        return pickle.load(f)

    def _load_documents_from_pickle(self):
        seg_id = self.rng.randint(0, 9)
        file_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/tokens/enwiki_train_tokens.{}"
        all_docs = []
        for j in range(100):
            full_id = seg_id * 100 + j
            f = open(file_path.format(full_id), "rb")
            all_docs.extend(pickle.load(f))
        return all_docs

    def create_instances_from_document(self, document):
        vocab_words = list(self.tokenizer.vocab.keys())
        max_num_tokens = self.max_seq_length - 2

        target_seq_length = max_num_tokens
        if self.rng.random() < self.short_seq_prob:
            target_seq_length = self.rng.randint(2, max_num_tokens)

        instances = []

        for raw_tokens in self.pool_tokens(document, target_seq_length):
            tokens, segment_ids = format_tokens_n_segid(raw_tokens)

            (tokens, masked_lm_positions,
             masked_lm_labels) = btd.create_masked_lm_predictions(tokens,
                                                                  self.masked_lm_prob,
                                                                  self.max_predictions_per_seq, vocab_words, self.rng)
            instance = TrainingInstance(
                tokens=tokens,
                segment_ids=segment_ids,
                is_random_next=False,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
            instances.append(instance)

        return instances

    def pad0(self, seq, max_len):
        assert len(seq) <= max_len
        while len(seq) < max_len:
            seq.append(0)
        return seq


class UnmaskedGen(LMTrainGen):
    def __init__(self):
        super(UnmaskedGen, self).__init__()

    def create_instances_from_document(self, document):
        max_num_tokens = self.max_seq_length - 2

        target_seq_length = max_num_tokens
        if self.rng.random() < self.short_seq_prob:
            target_seq_length = self.rng.randint(2, max_num_tokens)

        instances = []

        for raw_tokens in self.pool_tokens(document, target_seq_length):
            tokens, segment_ids = format_tokens_n_segid(raw_tokens)

            instance = SegmentInstance(
                tokens=tokens,
                segment_ids=segment_ids)
            instances.append(instance)

        return instances

    def write_instance_to_example_files(self, instances, output_files):
        """Create TF example files from `TrainingInstance`s."""
        writers = []
        for output_file in output_files:
            writers.append(tf.python_io.TFRecordWriter(output_file))

        writer_index = 0
        total_written = 0
        for (inst_index, instance) in enumerate(instances):
            input_ids = self.tokenizer.convert_tokens_to_ids(instance.tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = list(instance.segment_ids)

            max_seq_length = self.max_seq_length
            assert len(input_ids) <= self.max_seq_length
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            next_sentence_label = 1 if instance.is_random_next else 0

            features = collections.OrderedDict()
            features["input_ids"] = btd.create_int_feature(input_ids)
            features["input_mask"] = btd.create_int_feature(input_mask)
            features["segment_ids"] = btd.create_int_feature(segment_ids)
            features["next_sentence_labels"] = btd.create_int_feature([next_sentence_label])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))

            writers[writer_index].write(tf_example.SerializeToString())
            writer_index = (writer_index + 1) % len(writers)

            total_written += 1

            if inst_index < 20:
                log_print_inst(instance, features)

        for writer in writers:
            writer.close()

        tf_logging.info("Wrote %d total instances", total_written)


class UnmaskedPairGen(UnmaskedGen):
    def __init__(self):
        super(UnmaskedPairGen, self).__init__()

    def create_instances_from_document(self, document):
        raise Exception()

    def create_instances_from_documents(self, documents):
        max_num_tokens = self.max_seq_length - 3
        target_seq_length = max_num_tokens
        print("pooling chunks")
        docs_as_chunks, target_inst_num = self.pool_chunks_from_docs(documents, target_seq_length)
        print("num chunks : ", len(docs_as_chunks), target_inst_num)


        instances = []
        for _ in range(target_inst_num):
            chunk_1 = pick1(pick1(docs_as_chunks))

            m = self.rng.randint(1,len(chunk_1))
            tokens_a = flatten(chunk_1[:m])
            b_length = target_seq_length - len(tokens_a)
            if self.rng.random() < 0.5 :
                chunk_2 = pick1(pick1(docs_as_chunks))
                tokens_b = flatten(chunk_2)[:b_length]
            else:
                tokens_b = flatten(chunk_1[m:])[:b_length]
            truncate_seq_pair(tokens_a, tokens_b, target_seq_length, self.rng)

            tokens, segment_ids = format_tokens_pair_n_segid(tokens_a, tokens_b)
            instance = SegmentInstance(
                tokens=tokens,
                segment_ids=segment_ids)
            instances.append(instance)

        return instances


class MaskedPairGen(UnmaskedPairGen):
    def create_instances_from_documents(self, documents):
        instances = super(MaskedPairGen, self).create_instances_from_documents(documents)
        vocab_words = list(self.tokenizer.vocab.keys())

        inst_list = []
        for inst_index, inst in enumerate(instances):
            # We nedd both entries for entry prediction and LM prediction
            (tokens, masked_lm_positions,
             masked_lm_labels) = btd.create_masked_lm_predictions(inst.tokens,
                                                      self.masked_lm_prob,
                                                      self.max_predictions_per_seq, vocab_words, self.rng)
            instance = TrainingInstance(
                tokens=tokens,
                segment_ids=inst.segment_ids,
                is_random_next=False,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
            inst_list.append(instance)

            if inst_index < 20:
                tf_logging.info(instance.__str__())

        return inst_list

    def write_instances(self, new_inst_list, outfile):
        writer = RecordWriterWrap(outfile)
        example_numbers = []
        feature_formatter = MLMFeaturizer(self.tokenizer, self.max_seq_length, self.max_predictions_per_seq)

        for (inst_index, instance) in enumerate(new_inst_list):
            features = feature_formatter.instance_to_features(instance)
            writer.write_feature(features)
            if inst_index < 20:
                log_print_inst(instance, features)
        writer.close()

        tf_logging.info("Wrote %d total instances", writer.total_written)

        return example_numbers


class UnmaskedPairedDataGen(LMTrainBase):
    def __init__(self):
        super(UnmaskedPairedDataGen, self).__init__()

    def create_instances_from_documents(self, documents):
        documents = [doc for doc in documents if doc]
        max_num_tokens = self.max_seq_length - 3
        target_seq_length = max_num_tokens

        docs_as_chunks, target_inst_num = self.pool_chunks_from_docs(documents, target_seq_length)


        instances = []
        for _ in range(target_inst_num):
            chunk_1 = pick1(pick1(docs_as_chunks))

            m = self.rng.randint(1, len(chunk_1))
            tokens_a = flatten(chunk_1[:m])
            b_length = target_seq_length - len(tokens_a)
            if self.rng.random() < 0.5 :
                chunk_2 = pick1(pick1(docs_as_chunks))
                tokens_b = flatten(chunk_2)[:b_length]
            else:
                tokens_b = flatten(chunk_1[m:])[:b_length]
            truncate_seq_pair(tokens_a, tokens_b, target_seq_length, self.rng)

            tokens, segment_ids = format_tokens_pair_n_segid(tokens_a, tokens_b)
            instance = SegmentInstance(
                tokens=tokens,
                segment_ids=segment_ids)
            instances.append(instance)

        return instances

    def write_instances(self, new_inst_list, outfile):
        writer = RecordWriterWrap(outfile)
        example_numbers = []

        for (inst_index, instance) in enumerate(new_inst_list):
            features = get_basic_input_feature(self.tokenizer,
                                               self.max_seq_length,
                                               instance.tokens,
                                               instance.segment_ids)
            features["next_sentence_labels"] = btd.create_int_feature([0])

            writer.write_feature(features)
            if inst_index < 20:
                log_print_inst(instance, features)
        writer.close()

        tf_logging.info("Wrote %d total instances", writer.total_written)

        return example_numbers


class SegmentInstance(object):
    def __init__(self, tokens: object, segment_ids: object) -> object:
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = False

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                             is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
                [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
                [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
                [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def pool_tokens(max_seq_length, rng, document, target_seq_length, skip = False):
    results = []
    current_chunk = []
    current_length = 0
    max_num_tokens = max_seq_length - 2
    i = 0
    if skip:
        i = i + rng.randint(0,3)
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            tokens_a = flatten(current_chunk)
            tokens_a = truncate_seq(tokens_a, max_num_tokens, rng)
            results.append(tokens_a)
            current_chunk = []
            current_length = 0
            if skip:
                i = i + rng.randint(0, 3)
        i += 1
    return results


def get_masked_lm_features_as_list(tokenizer, max_predictions_per_seq, masked_lm_positions, masked_lm_labels):
    masked_lm_positions = list(masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)
    return masked_lm_positions, masked_lm_ids, masked_lm_weights


def get_masked_lm_features(tokenizer, max_predictions_per_seq, masked_lm_positions, masked_lm_labels):
    masked_lm_positions, masked_lm_ids, masked_lm_weights = \
        get_masked_lm_features_as_list(tokenizer, max_predictions_per_seq, masked_lm_positions, masked_lm_labels)

    features = collections.OrderedDict()
    features["masked_lm_positions"] = btd.create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = btd.create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = btd.create_float_feature(masked_lm_weights)
    return features