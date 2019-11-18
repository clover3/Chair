from tlm.data_gen.base import LMTrainGen, UnmaskedPairGen
from data_generator import tokenizer_wo_tf as tokenization
from tlm.wiki import bert_training_data as btd
import tensorflow as tf
from tlm.tf_logging import tf_logging
import collections
from misc_lib import pick1, TimeEstimator
from models.classic.stopword import load_stopwords

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


class DictTrainGen(UnmaskedPairGen):
    def __init__(self, dictionary):
        super(DictTrainGen, self).__init__()
        self.dict = dictionary
        self.only_sinlge_def = True
        self.max_def_length = 128
        self.max_d_loc = 16
        self.stopword = load_stopwords()
        self.d_mask_token = "[unused4]"

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
            elif len(w.subword_rep) == 1:
                pass
            else:
                candidate.append(w)
        if candidate:
            return pick1(candidate)
        else:
            return None

    def get_word_tokens(self, tokens):
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

    def hide_word(self, tokens, target_word):
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
        for idx in locations:
            tokens[idx] = self.d_mask_token
        assert locations
        return tokens, locations

    def create_instances_from_documents(self, documents):
        instances = super(DictTrainGen, self).create_instances_from_documents(documents)

        new_inst_list = []
        cnt = 0
        for inst in instances:
            words = self.get_word_tokens(inst.tokens)
            selected_word = self.draw_word(words)
            if selected_word is not None:
                tokens, locations = self.hide_word(inst.tokens, selected_word)
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

            d_input_ids = self.pad0(d_input_ids, self.max_def_length)
            d_input_mask = self.pad0(d_input_mask, self.max_def_length)
            d_location_ids = self.pad0(instance.word_loc_list[:self.max_d_loc], self.max_d_loc)

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
        tf_logging.info("cnt_def_overlen: %d", cnt_def_overlen)
        tf_logging.info("multi_sb: %d", multi_sb)
        tf_logging.info("None cnt: %d", cnt_none)
