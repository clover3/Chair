from tlm.data_gen.base import LMTrainGen, UnmaskedGen
from data_generator import tokenizer_wo_tf as tokenization
from tlm.wiki import bert_training_data as btd
import tensorflow as tf
from tlm.tf_logging import tf_logging
import collections


class SegmentInstanceWithDictEntry(object):
    def __init__(self, tokens, segment_ids, dict_word, dict_def, word_loc_list):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = False
        self.dict_word = dict_word
        self.dict_def = dict_def
        self.word_loc_list = word_loc_list

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class Dictionary:

    def __init__(self):
        NotImplemented


    # Input : word_as_sw : subword tokens
    # Output : dictionary definition as subword tokens
    def lookup(self, word_as_sw):
        NotImplemented

class DictTrainGen(UnmaskedGen):
    def __init__(self, dictionary):
        super(DictTrainGen, self).__init__()
        self.dict = dictionary
        self.only_sinlge_def = True
        self.max_def_length = 128
        self.max_d_loc = 16

    def draw_word(self, words):
        # exclude stop words
        # include if it is in dictionary
        # uniformly sample among remaining
        NotImplemented

    def get_word_tokens(self, tokens):
        NotImplemented

    def hide_word(self, tokens, target_word):
        NotImplemented

    def create_instances_from_document(self, document):
        instances = super(DictTrainGen, self).create_instances_from_document(document)

        new_inst_list = []
        for inst in instances:
            words = self.get_word_tokens(inst.tokens)
            selected_word = self.draw_word(words)
            tokens, locations = self.hide_word(inst.tokens, selected_word)
            new_inst = SegmentInstanceWithDictEntry(
                tokens,
                inst.segment_ids,
                selected_word,
                self.dict.lookup(selected_word),
                locations
            )
            new_inst_list.append(new_inst)

        return new_inst_list

    def write_instance_to_example_files(self, instances, output_files):
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

            d_input_ids = self.tokenizer.convert_tokens_to_ids(instance.dict_def)
            d_input_mask = [1] * len(d_input_ids)

            d_input_ids = self.pad0(d_input_ids, self.max_def_length)
            d_input_mask = self.pad0(d_input_mask, self.max_def_length)
            d_location_ids = self.pad0(instance.location, self.max_d_loc)

            next_sentence_label = 1 if instance.is_random_next else 0

            features = collections.OrderedDict()
            features["input_ids"] = btd.create_int_feature(input_ids)
            features["input_mask"] = btd.create_int_feature(input_mask)
            features["segment_ids"] = btd.create_int_feature(segment_ids)
            features["d_input_ids"] = btd.create_int_feature(d_input_ids)
            features["d_input_mask"] = btd.create_int_feature(d_input_mask)
            features["d_location_ids"] = btd.create_int_feature(d_location_ids)

            features["next_sentence_labels"] = btd.create_int_feature([next_sentence_label])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))

            writers[writer_index].write(tf_example.SerializeToString())
            writer_index = (writer_index + 1) % len(writers)

            total_written += 1

            if inst_index < 20:
                self.log_print_inst(instance, features)

        for writer in writers:
            writer.close()

        tf_logging.info("Wrote %d total instances", total_written)
