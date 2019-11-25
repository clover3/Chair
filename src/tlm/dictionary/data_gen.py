from tlm.data_gen.base import LMTrainGen, UnmaskedPairGen
from data_generator import tokenizer_wo_tf as tokenization
import tensorflow as tf
from tlm.tf_logging import tf_logging
import collections
from misc_lib import pick1, TimeEstimator, average, lmap
from models.classic.stopword import load_stopwords
import tlm.data_gen.bert_data_gen as btd
import random
import os
from collections import Counter
from path import data_path
from trainer.tf_module import get_batches_ex

class Word:
    def __init__(self, subword_tokens):
        self.subword_rep = subword_tokens
        self.word = subword_tokens[0]
        for sw in subword_tokens[1:]:
            assert sw[:2] == "##"
            self.word += sw[2:]

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
        s += "dict_word: %s\n" % (" ".join([str(x) for x in self.dict_word.word]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


TOKEN_LINE_SEP = "[unused5]"
TOKEN_DEF_SEP = "[unused6]"


class DictPredictionEntry(object):
    def __init__(self, tokens, segment_ids, dict_word, dict_def, word_loc_list, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = False
        self.dict_word = dict_word
        self.dict_def = dict_def
        self.word_loc_list = word_loc_list
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "dict_word: %s\n" % (" ".join([str(x) for x in self.dict_word.word]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def dictionary_encoder(entries, tokenizer):
    # entry = {
    #         'word': word,
    #         'content': content_list,
    #         'head': head_list,
    #         }
    result_dict = {}

    def encode_content(content_list):
        all_tokens = []
        for line in content_list:
            tokens = tokenizer.tokenize(line)
            all_tokens.extend(tokens)
            all_tokens.append(TOKEN_LINE_SEP)
        return all_tokens

    ticker = TimeEstimator(len(entries))
    for e in entries:
        word = e['word'].lower()
        content = encode_content(e['content'])
        if word in result_dict:
            pre_content = result_dict[word]
            result_dict[word] = pre_content + [TOKEN_DEF_SEP] + content
        else:
            result_dict[word] = content
        ticker.tick()
    return result_dict

class Dictionary:
    def __init__(self, word_to_dict_tokens):
        self.d = word_to_dict_tokens

    # Input : Word object
    # Output : dictionary definition as subword tokens
    def lookup(self, word):
        return self.d[word]

    def __contains__(self, word):
        return word in self.d


def is_continuation(subword):
    return len(subword) > 2 and subword[:2] == "##"


def get_word_tokens(tokens):
    words = []
    cur_word = []
    for subword in tokens:
        if is_continuation(subword):
            cur_word.append(subword)
        else:
            if cur_word:
                words.append(Word(cur_word))
            cur_word = [subword]
    return words


def filter_unique_words(words):
    words_set = set()
    output_list = []
    for word in words:
        if word.word not in words_set:
            words_set.add(word.word)
            output_list.append(word)
    return output_list


def get_locations(tokens, target_word):
    t_idx = 0
    locations = []
    sw_len = len(target_word.subword_rep)
    try:
        for st_idx, t in enumerate(tokens):
            match = True
            for t_idx in range(sw_len):
                if not tokens[st_idx+t_idx] == target_word.subword_rep[t_idx]:
                    match = False
                    break

            if match:
                if not is_continuation(tokens[st_idx + sw_len]):
                    for j in range(st_idx, st_idx+sw_len):
                        locations.append(j)

    except IndexError as e:
        print(target_word.subword_rep)
        print(tokens)
        raise e

    return locations


def pad0(seq, max_len):
    assert len(seq) <= max_len
    while len(seq) < max_len:
        seq.append(0)
    return seq


class DictAugment:
    def __init__(self, data, dictionary):
        self.max_def_length = 256
        self.max_d_loc = 16
        self.stopword = load_stopwords()

        self.data_info = {}
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.dict = self.encode_dict_as_feature(dictionary)
        self.data = data

        ticker = TimeEstimator(len(data), "nli indexing", 100)
        for data_idx, e in enumerate(data):
            input_ids, input_mask, segment_ids, y = e
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            self.data_info[data_idx] = self.index(tokens)
            ticker.tick()

        self.n_data = len(data)

    def encode_dict_as_feature(self, dictionary):
        new_dict = {}
        for word, dict_def in dictionary.items():
            d_input_ids = self.tokenizer.convert_tokens_to_ids(dict_def)
            d_input_ids = d_input_ids[:self.max_def_length]
            d_input_mask = [1] * len(d_input_ids)
            d_input_ids = pad0(d_input_ids, self.max_def_length)
            d_input_mask = pad0(d_input_mask, self.max_def_length)
            new_dict[word] = d_input_ids, d_input_mask
        return new_dict

    def index(self, tokens):
        words = get_word_tokens(tokens)
        unique_words = filter_unique_words(words)
        valid_words = []
        for word in unique_words:
            if word.word in self.stopword:
                pass
            elif word.word not in self.dict:
                pass
            else:
                word.location = get_locations(tokens, word)
                word.location = word.location[:self.max_d_loc]
                word.enc_location = pad0(word.location, self.max_d_loc)
                valid_words.append(word)

        return valid_words

    def get_best_def(self, data_idx, ranked_locations):
        location_to_word = {}
        for word in self.data_info[data_idx]:
            for loc in word.location:
                location_to_word[loc] = word

        for loc in ranked_locations:
            if loc in location_to_word:
                return location_to_word[loc]

        return None

    def get_random_batch(self, batch_size):
        data = []
        for _ in range(batch_size):
            data_idx = random.randint(0, self.n_data - 1)
            input_ids, input_mask, segment_ids, y = self.data[data_idx]
            appeared_words = self.data_info[data_idx]
            if appeared_words:
                word = pick1(appeared_words)
                d_input_ids, d_input_mask = self.dict[word.word]
                d_location_ids = word.location
            else:
                d_input_ids = [0] * self.max_def_length
                d_input_mask = [0] * self.max_def_length
                d_location_ids = [0] * self.max_d_loc

            e = input_ids, input_mask, segment_ids, d_input_ids, d_input_mask, d_location_ids, y
            data.append(e)
        return get_batches_ex(data, batch_size, 7)[0]


def hide_word(tokens, target_word, d_mask_token):
    new_tokens = list(tokens)
    locations = get_locations(new_tokens, target_word)
    for idx in locations:
        new_tokens[idx] = d_mask_token
    assert locations
    return new_tokens, locations


class DictTrainGen(UnmaskedPairGen):
    def __init__(self, dictionary):
        super(DictTrainGen, self).__init__()
        self.dict = dictionary
        self.max_def_length = 256
        self.max_d_loc = 16
        self.stopword = load_stopwords()
        self.d_mask_token = "[unused4]"
        self.f_hide_word = True
        self.max_word_len = 8
        self.drop_none_dict = False
        self.no_dict_assist = False
        self.drop_short_word = True

    def draw_word(self, words):
        # exclude stop words
        # include if it is in dictionary
        # uniformly sample among remaining
        candidate = []
        for w in words:
            if w.word in self.stopword:
                pass
            elif w.word not in self.dict:
                pass
            elif self.drop_short_word and len(w.subword_rep) == 1:
                pass
            else:
                candidate.append(w)
        if candidate:
            return pick1(candidate)
        else:
            return None



    def create_instances_from_documents(self, documents):
        instances = super(DictTrainGen, self).create_instances_from_documents(documents)

        new_inst_list = []
        cnt = 0
        for inst in instances:
            if not self.no_dict_assist:
                words = get_word_tokens(inst.tokens)
                selected_word = self.draw_word(words)
            else:
                selected_word = None
            if selected_word is not None:
                if self.f_hide_word :
                    tokens, locations = hide_word(inst.tokens, selected_word, self.d_mask_token)
                else:
                    locations = get_locations(inst.tokens, selected_word)
                    tokens = inst.tokens
                new_inst = SegmentInstanceWithDictEntry(
                    tokens,
                    inst.segment_ids,
                    selected_word,
                    self.dict.lookup(selected_word.word),
                    locations
                )

            else:
                new_inst = SegmentInstanceWithDictEntry(
                    inst.tokens,
                    inst.segment_ids,
                    selected_word,
                    [],
                    []
                )

            new_inst_list.append(new_inst)

            if cnt < 20:
                tf_logging.info("Example Instance:")
                tf_logging.info("Tokens : {}".format(new_inst.tokens))
                tf_logging.info("Text : {}".format(tokenization.pretty_tokens(new_inst.tokens)))
                if new_inst.dict_word is not None:
                    tf_logging.info("selected_word : {}".format(new_inst.dict_word.word))
                else:
                    tf_logging.info("No word selected")
                tf_logging.info("dict_def : {}".format(new_inst.dict_def))
                tf_logging.info("word loc : {}".format(new_inst.word_loc_list))
                tf_logging.info("-------------------")
                cnt += 1
        return new_inst_list

    def write_instance_to_example_files(self, instances, output_files):
        writers = []
        for output_file in output_files:
            writers.append(tf.python_io.TFRecordWriter(output_file))

        cnt_def_overlen = 0
        multi_sb = 0
        cnt_none = 0

        writer_index = 0
        total_written = 0
        for (inst_index, instance) in enumerate(instances):
            if self.drop_none_dict and instance.dict_word is None:
                continue

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
            if len(d_input_ids) > self.max_def_length:
                cnt_def_overlen += 1

            if instance.dict_word is None:
                cnt_none += 1
            else:
                if len(instance.dict_word.subword_rep) > 1 :
                    multi_sb += 1

            d_input_ids = d_input_ids[:self.max_def_length]
            d_input_mask = [1] * len(d_input_ids)

            if instance.word_loc_list:
                target_segment = segment_ids[instance.word_loc_list[0]]
                d_segment_ids = [target_segment] * len(d_input_ids)
            else:
                d_segment_ids = []

            if instance.dict_word is not None:
                selected_word = self.tokenizer.convert_tokens_to_ids(instance.dict_word.subword_rep)
            else:
                selected_word = []

            d_input_ids = self.pad0(d_input_ids, self.max_def_length)
            d_input_mask = self.pad0(d_input_mask, self.max_def_length)
            d_location_ids = self.pad0(instance.word_loc_list[:self.max_d_loc], self.max_d_loc)
            d_segment_ids = self.pad0(d_segment_ids, self.max_def_length)
            selected_word = self.pad0(selected_word, self.max_word_len)

            next_sentence_label = 1 if instance.is_random_next else 0

            features = collections.OrderedDict()
            features["input_ids"] = btd.create_int_feature(input_ids)
            features["input_mask"] = btd.create_int_feature(input_mask)
            features["segment_ids"] = btd.create_int_feature(segment_ids)
            features["d_input_ids"] = btd.create_int_feature(d_input_ids)
            features["d_input_mask"] = btd.create_int_feature(d_input_mask)
            features["d_segment_ids"] = btd.create_int_feature(d_segment_ids)
            features["d_location_ids"] = btd.create_int_feature(d_location_ids)
            features["selected_word"] = btd.create_int_feature(selected_word)
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
        tf_logging.info("cnt_def_overlen: %d", cnt_def_overlen)
        tf_logging.info("multi_sb: %d", multi_sb)
        tf_logging.info("None cnt: %d", cnt_none)


class DictLookupPredictGen(DictTrainGen):
    def __init__(self, dictionary, samples_n):
        super(DictTrainGen, self).__init__()
        self.dict = dictionary
        self.max_def_length = 256
        self.max_d_loc = 16
        self.stopword = load_stopwords()
        self.d_mask_token = "[unused4]"
        self.max_word_len = 8
        self.samples_n = samples_n

        self.event_counter = Counter()


    def make_tf_example(self, instance):
        max_seq_length = self.max_seq_length
        input_ids = self.tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = self.tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) <  self.max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = btd.create_int_feature(input_ids)
        features["input_mask"] = btd.create_int_feature(input_mask)
        features["segment_ids"] = btd.create_int_feature(segment_ids)
        features["masked_lm_positions"] = btd.create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = btd.create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = btd.create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = btd.create_int_feature([next_sentence_label])

        return tf.train.Example(features=tf.train.Features(feature=features))

    def create_instances_from_documents(self, documents):
        instances = super(DictTrainGen, self).create_instances_from_documents(documents)
        vocab_words = list(self.tokenizer.vocab.keys())

        def valid_candidate(w):
            return w.word not in self.stopword and w.word in self.dict


        new_inst_list = []
        cnt = 0
        for inst in instances:
            (tokens, masked_lm_positions,
             masked_lm_labels) = btd.create_masked_lm_predictions(inst.tokens,
                                                      self.masked_lm_prob,
                                                      self.max_predictions_per_seq, vocab_words, self.rng)
            key_inst = btd.TrainingInstance(inst.tokens, inst.segment_ids, masked_lm_positions, masked_lm_labels, False)
            words = get_word_tokens(tokens)
            words = filter_unique_words(words)
            words = list([w for w in words if valid_candidate(w)])
            random.shuffle(words)
            words = words[:self.samples_n]


            examples = []

            sub_inst = DictPredictionEntry(
                tokens,
                inst.segment_ids,
                None,
                [],
                [],
                masked_lm_positions,
                masked_lm_labels
            )
            examples.append(sub_inst)

            for selected_word in words:
                assert selected_word is not None
                hidden_tokens, locations = hide_word(tokens, selected_word, self.d_mask_token)
                sub_inst = DictPredictionEntry(
                    hidden_tokens,
                    inst.segment_ids,
                    selected_word,
                    self.dict.lookup(selected_word.word),
                    locations,
                    masked_lm_positions,
                    masked_lm_labels
                )
                examples.append(sub_inst)

            new_inst_list.append((key_inst , examples))

            if cnt < 20:
                tf_logging.info("Example Instance:")
                tf_logging.info("Tokens : {}".format(inst.tokens))
                tf_logging.info("Text : {}".format(tokenization.pretty_tokens(inst.tokens)))

                w_str = ""
                for inst in examples:
                    if inst.dict_word is not None:
                        w_str += " " + inst.dict_word.word

                tf_logging.info("words : {}".format(w_str))
                tf_logging.info("-------------------")
                cnt += 1
        return new_inst_list

    def get_features_from_instance(self, instance):
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
        if len(d_input_ids) > self.max_def_length:
            self.event_counter["def_overlen"] += 1

        if instance.dict_word is None:
            self.event_counter["None"] += 1
        else:
            if len(instance.dict_word.subword_rep) > 1:
                self.event_counter["multi_subword"] += 1

        d_input_ids = d_input_ids[:self.max_def_length]
        d_input_mask = [1] * len(d_input_ids)

        if instance.word_loc_list:
            target_segment = segment_ids[instance.word_loc_list[0]]
            d_segment_ids = [target_segment] * len(d_input_ids)
        else:
            d_segment_ids = []

        if instance.dict_word is not None:
            selected_word = self.tokenizer.convert_tokens_to_ids(instance.dict_word.subword_rep)
        else:
            selected_word = []

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = self.tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < self.max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        d_input_ids = self.pad0(d_input_ids, self.max_def_length)
        d_input_mask = self.pad0(d_input_mask, self.max_def_length)
        d_location_ids = self.pad0(instance.word_loc_list[:self.max_d_loc], self.max_d_loc)
        d_segment_ids = self.pad0(d_segment_ids, self.max_def_length)
        selected_word = self.pad0(selected_word, self.max_word_len)

        next_sentence_label = 1 if instance.is_random_next else 0
        features = collections.OrderedDict()
        features["input_ids"] = btd.create_int_feature(input_ids)
        features["input_mask"] = btd.create_int_feature(input_mask)
        features["segment_ids"] = btd.create_int_feature(segment_ids)
        features["masked_lm_positions"] = btd.create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = btd.create_int_feature(masked_lm_ids)
        features["d_input_ids"] = btd.create_int_feature(d_input_ids)
        features["d_input_mask"] = btd.create_int_feature(d_input_mask)
        features["d_segment_ids"] = btd.create_int_feature(d_segment_ids)
        features["d_location_ids"] = btd.create_int_feature(d_location_ids)
        features["selected_word"] = btd.create_int_feature(selected_word)
        features["next_sentence_labels"] = btd.create_int_feature([next_sentence_label])
        return features

    def reset_event_counter(self):
        self.event_counter = Counter()

    def write_instances(self, instances, output_file, key_output_file):
        writer = tf.python_io.TFRecordWriter(output_file)

        self.reset_event_counter()
        total_written = 0
        key_writer = tf.python_io.TFRecordWriter(key_output_file)

        n_example_list = []
        for (inst_index, (key_inst, examples)) in enumerate(instances):
            n_example_list.append(len(examples))
            for instance in examples:
                features = self.get_features_from_instance(instance)
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
                total_written += 1

            tf_example = self.make_tf_example(key_inst)
            key_writer.write(tf_example.SerializeToString())
        writer.close()
        key_writer.close()

        tf_logging.info("Average example per inst : {}".format(average(n_example_list)))
        tf_logging.info("Wrote %d total instances", total_written)
        for key, value in self.event_counter.items():
            tf_logging.info("Count of {}: {}".format(key, value))

        return n_example_list



class DictEntryPredictGen(UnmaskedPairGen):
    def __init__(self, parsed_dictionary, max_def_entry):
        super(DictEntryPredictGen, self).__init__()
        self.parsed_dictionary = parsed_dictionary
        self.max_def_length = 256
        self.max_d_loc = 16
        self.stopword = load_stopwords()
        self.d_mask_token = "[unused4]"
        self.max_word_len = 8
        self.max_def_entry = max_def_entry

        self.event_counter = Counter()


    def create_instances_from_documents(self, documents):
        instances = super(DictEntryPredictGen, self).create_instances_from_documents(documents)
        vocab_words = list(self.tokenizer.vocab.keys())

        def valid_candidate(w):
            return w.word not in self.stopword and w.word in self.parsed_dictionary


        new_inst_list = []
        cnt = 0
        for inst in instances:
            # We nedd both entries for entry prediction and LM prediction
            (tokens, masked_lm_positions,
             masked_lm_labels) = btd.create_masked_lm_predictions(inst.tokens,
                                                      self.masked_lm_prob,
                                                      self.max_predictions_per_seq, vocab_words, self.rng)
            key_inst = SegmentInstanceWithDictEntry(
                    tokens,
                    inst.segment_ids,
                    None,
                    [],
                    []
                )

            words = get_word_tokens(tokens)
            words = list([w for w in words if valid_candidate(w)])
            def get_num_defs(word):
                return len(self.parsed_dictionary[word.word])

            multi_def_words = list([w for w in words if get_num_defs(w) > 1])
            if not multi_def_words:
                tf_logging.debug("Continue because no multi sense word founds")
                continue

            selected_word = pick1(multi_def_words)
            hidden_tokens, locations = hide_word(tokens, selected_word, self.d_mask_token)

            examples = []
            for word_def in self.parsed_dictionary[selected_word.word]:
                sub_inst = SegmentInstanceWithDictEntry(
                    hidden_tokens,
                    inst.segment_ids,
                    selected_word,
                    ["[CLS]"] + word_def,
                    locations
                )
                examples.append(sub_inst)

            new_inst_list.append((key_inst, examples))

            if cnt < 20:
                tf_logging.info("Example Instance:")
                tf_logging.info("Tokens : {}".format(inst.tokens))
                tf_logging.info("Text : {}".format(tokenization.pretty_tokens(inst.tokens)))

                for inst in examples:
                    if inst.dict_def is not None:
                        def_str = tokenization.pretty_tokens((inst.dict_def))
                        tf_logging.info("Definitions: {}".format(def_str))
                tf_logging.info("-------------------")
                cnt += 1
        return new_inst_list

    def write_instances(self, new_inst_list, outfile):
        writer = tf.python_io.TFRecordWriter(outfile)
        cnt_def_overlen = 0
        multi_sb = 0
        cnt_none = 0
        example_numbers = []

        total_written = 0
        for (inst_index, new_inst) in enumerate(new_inst_list):
            key_inst, examples = new_inst
            example_numbers.append(len(examples)+1)
            for instance in [key_inst] + examples:
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
                if len(d_input_ids) > self.max_def_length:
                    cnt_def_overlen += 1

                if instance.dict_word is None:
                    cnt_none += 1
                else:
                    if len(instance.dict_word.subword_rep) > 1 :
                        multi_sb += 1

                d_input_ids = d_input_ids[:self.max_def_length]
                d_input_mask = [1] * len(d_input_ids)

                if instance.word_loc_list:
                    target_segment = segment_ids[instance.word_loc_list[0]]
                    d_segment_ids = [target_segment] * len(d_input_ids)
                else:
                    d_segment_ids = []

                if instance.dict_word is not None:
                    selected_word = self.tokenizer.convert_tokens_to_ids(instance.dict_word.subword_rep)
                else:
                    selected_word = []

                d_input_ids = self.pad0(d_input_ids, self.max_def_length)
                d_input_mask = self.pad0(d_input_mask, self.max_def_length)
                d_location_ids = self.pad0(instance.word_loc_list[:self.max_d_loc], self.max_d_loc)
                d_segment_ids = self.pad0(d_segment_ids, self.max_def_length)
                selected_word = self.pad0(selected_word, self.max_word_len)

                next_sentence_label = 1 if instance.is_random_next else 0

                features = collections.OrderedDict()
                features["input_ids"] = btd.create_int_feature(input_ids)
                features["input_mask"] = btd.create_int_feature(input_mask)
                features["segment_ids"] = btd.create_int_feature(segment_ids)
                features["d_input_ids"] = btd.create_int_feature(d_input_ids)
                features["d_input_mask"] = btd.create_int_feature(d_input_mask)
                features["d_segment_ids"] = btd.create_int_feature(d_segment_ids)
                features["d_location_ids"] = btd.create_int_feature(d_location_ids)
                features["selected_word"] = btd.create_int_feature(selected_word)
                features["next_sentence_labels"] = btd.create_int_feature([next_sentence_label])

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))

                writer.write(tf_example.SerializeToString())

                total_written += 1

                if inst_index < 20:
                    self.log_print_inst(instance, features)

        writer.close()

        tf_logging.info("Wrote %d total instances", total_written)
        tf_logging.info("cnt_def_overlen: %d", cnt_def_overlen)
        return example_numbers
