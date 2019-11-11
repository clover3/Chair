import os
import pickle
import random
from path import data_path
import tensorflow as tf
import collections

from data_generator import tokenizer_wo_tf as tokenization
from misc_lib import flatten
from tlm.wiki import bert_training_data as btd
from tlm.tf_logging import logging

def truncate_seq(tokens_a, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a)
        if total_length <= max_num_tokens:
            break

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del tokens_a[0]
        else:
            tokens_a.pop()
    return tokens_a


class LMTrainGen:
    def __init__(self):
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

        self.masked_lm_prob = 0.15
        self.short_seq_prob = 0.1
        self.problem_per_job = 100 * 1000
        self.max_seq_length = 512
        self.max_predictions_per_seq = 20
        self.dupe_factor = 1
        self.rng = random.Random(1)

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

    def pool_tokens(self, document, target_seq_length):
        results = []
        current_chunk = []
        current_length = 0
        max_num_tokens = self.max_seq_length - 2
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                tokens_a = flatten(current_chunk)
                tokens_a = truncate_seq(tokens_a, max_num_tokens, self.rng)
                results.append(tokens_a)
                current_chunk = []
                current_length = 0
            i += 1
        return results

    def format_tokens_n_segid(self, raw_tokens):
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in raw_tokens:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)
        return tokens, segment_ids

    def create_instances_from_document(self, document):
        vocab_words = list(self.tokenizer.vocab.keys())
        max_num_tokens = self.max_seq_length - 2

        target_seq_length = max_num_tokens
        if self.rng.random() < self.short_seq_prob:
            target_seq_length = self.rng.randint(2, max_num_tokens)

        instances = []

        for raw_tokens in self.pool_tokens(document, target_seq_length):
            tokens, segment_ids = self.format_tokens_n_segid(raw_tokens)

            (tokens, masked_lm_positions,
             masked_lm_labels) = btd.create_masked_lm_predictions(tokens,
                                                                  self.masked_lm_prob,
                                                                  self.max_predictions_per_seq, vocab_words, self.rng)
            instance = btd.TrainingInstance(
                tokens=tokens,
                segment_ids=segment_ids,
                is_random_next=False,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
            instances.append(instance)

        return instances

class SegmentInstance(object):
    def __init__(self, tokens, segment_ids):
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
            tokens, segment_ids = self.format_tokens_n_segid(raw_tokens)

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
                logging.info("*** Example ***")
                logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in instance.tokens]))

                for feature_name in features.keys():
                    feature = features[feature_name]
                    values = []
                    if feature.int64_list.value:
                        values = feature.int64_list.value
                    elif feature.float_list.value:
                        values = feature.float_list.value
                    logging.info(
                        "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

        for writer in writers:
            writer.close()

        logging.info("Wrote %d total instances", total_written)
