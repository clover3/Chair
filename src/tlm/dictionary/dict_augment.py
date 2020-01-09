import os
import random
from abc import ABC, abstractmethod

import numpy as np

from cache import load_cache, save_to_pickle, load_from_pickle
from cpath import data_path
from data_generator import tokenizer_wo_tf as tokenization
from misc_lib import TimeEstimator, pick1
from models.classic.stopword import load_stopwords
from tf_util.tf_logging import tf_logging
from tlm.data_gen.base import pad0
from tlm.dictionary.data_gen import get_word_tokens, filter_unique_words, get_locations
from tlm.model_cnfig import JsonConfig
from tlm.training.train_flags import FLAGS
from trainer.np_modules import get_batches_ex

NAME_WSSDR = "wssdr"
NAME_dict_1 = "dict_1"
NAME_DUMMY_WSSDR = "dummy_wssdr"
NAME_APR = "apr"

class DictAugmentedDataLoader:
    def __init__(self, feeder_name, data_loader, use_cache=False):
        self.use_cache = use_cache
        self.data_loader = data_loader
        if feeder_name == NAME_dict_1:
            self.feeder_getter = self.get_dict1_feeder
        elif feeder_name == NAME_WSSDR:
            self.feeder_getter = self.get_wssdr_feeder
        elif feeder_name == NAME_APR:
            self.feeder_getter = self.get_wssdr_feeder
        elif feeder_name == NAME_DUMMY_WSSDR:
            self.feeder_getter = self.get_dummy_wssdr_feeder
        else:
            raise Exception()

        self.feeder_name = feeder_name

    def _load_data_feeder(self, split_name, original_data_getter):
        original_data = original_data_getter()
        cache_name = split_name + "_feeder." + self.feeder_name
        if self.use_cache:
            data_feeder = load_cache(cache_name)
        else:
            data_feeder = None


        if data_feeder is None:
            tf_logging.info("Parsing terms in " + split_name)
            data_feeder = self.feeder_getter(original_data)
            save_to_pickle(data_feeder, cache_name)
        return data_feeder

    @staticmethod
    def get_wssdr_feeder(data):
        dictionary = load_from_pickle("webster_parsed_w_cls")
        ssdr_config = JsonConfig.from_json_file(FLAGS.model_config_file)
        return SSDRAugment(data, dictionary, ssdr_config, FLAGS.def_per_batch, FLAGS.max_def_length)

    @staticmethod
    def get_dummy_wssdr_feeder(data):
        dictionary = {}
        ssdr_config = JsonConfig.from_json_file(FLAGS.model_config_file)
        return SSDRAugment(data, dictionary, ssdr_config, FLAGS.def_per_batch, FLAGS.max_def_length)

    @staticmethod
    def get_dict1_feeder(data):
        dictionary = load_from_pickle("webster")
        return DictAugment(data, dictionary)

    def get_train_feeder(self):
        return self._load_data_feeder("train_data", self.data_loader.get_train_data)

    def get_dev_feeder(self):
        return self._load_data_feeder("dev_data", self.data_loader.get_dev_data)


class DictAuxDataFeederInterface(ABC):
    def __init__(self, data):
        self.data = data
        self.data_len = len(self.data)

    def get_data_len(self):
        return self.data_len

    @abstractmethod
    def get_random_batch(self, batch_size):
        pass

    @abstractmethod
    def get_lookup_batch(self, batch_size):
        pass

class DictAuxDataFeeder(DictAuxDataFeederInterface):
    def __init__(self, data, data_info):
        super(DictAuxDataFeeder, self).__init__(data)
        self.stopword = load_stopwords()
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

        self.dict = self.encode_dict_as_feature(self.raw_dictionary)

        # data is already truncated and padded
        self.data = data

        self.data_len = len(self.data)
        if data_info is not None:
            self.data_info = data_info
        else:
            self.data_info = self.nli_data_indexing(data)

    def nli_data_indexing(self, data):
        data_info = {}
        ticker = TimeEstimator(len(data), "nli indexing", 100)
        for data_idx, e in enumerate(data):
            input_ids, input_mask, segment_ids, y = e
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            data_info[data_idx] = self.index(tokens)
            ticker.tick()
        return data_info

    @abstractmethod
    def encode_dict_as_feature(self, dictionary):
        pass

    def get_data_len(self):
        return self.data_len

    def index(self, tokens):
        words = get_word_tokens(tokens)
        unique_words = filter_unique_words(words)
        valid_words = []
        for word in unique_words:
            if word.word in self.stopword:
                pass
            elif not self.dict_contains(word.word):
                pass
            else:
                word.location = get_locations(tokens, word)
                word.location = word.location[:self.get_max_d_loc()]
                word.enc_location = pad0(word.location, self.get_max_d_loc())
                valid_words.append(word)

        return valid_words

    @staticmethod
    def get_word_from_rank_list(ranked_locations, location_to_word):
        for loc in ranked_locations:
            if loc in location_to_word:
                return location_to_word[loc]
        return None

    def select_best_lookup_term(self, data_idx, ranked_locations):
        words = self.data_info[data_idx]
        location_to_word = self.invert_index_word_locations(words)
        selected_word = self.get_word_from_rank_list(ranked_locations, location_to_word)
        return selected_word

    @staticmethod
    def invert_index_word_locations(words):
        location_to_word = {}
        for word in words:
            for loc in word.location:
                location_to_word[loc] = word
        return location_to_word

    @abstractmethod
    def get_max_d_loc(self):
        pass

    @abstractmethod
    def dict_contains(self, word: str): #
        pass


class DictAugment(DictAuxDataFeeder):
    def __init__(self, data, dictionary):
        self.max_def_length = 256
        self.max_d_loc = 16
        self.raw_dictionary = dictionary
        super(DictAugment, self).__init__(data)

    def get_max_d_loc(self):
        return self.max_d_loc

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

    def dict_contains(self, word: str):
        return word in self.dict

    def get_random_batch(self, batch_size):
        data = []
        for _ in range(batch_size):
            data_idx = random.randint(0, self.data - 1)
            input_ids, input_mask, segment_ids, y = self.data[data_idx]
            appeared_words = self.data_info[data_idx]
            if appeared_words :
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


class LookupTrainInfo:
    def __init__(self):
        self.n_groups = 0
        self.dict_num_candidates = {}
        self.flat_instances = []

    def add_group(self, entries):
        self.dict_num_candidates[self.n_groups] = len(entries)
        self.n_groups += 1
        self.flat_instances.extend(entries)

    def select_best(self, losses):
        local_best_indices = self._get_best_indices_locally(losses)
        selected_instances = []
        offset = 0
        for group_idx in range(self.n_groups):

            local_idx = local_best_indices[group_idx]
            if local_idx >= 0:
                global_idx = offset + local_best_indices[group_idx]
                selected_instances.append(self.flat_instances[global_idx])

        return selected_instances

    def _get_best_indices_locally(self, losses):
        local_best_indices = []
        assert len(losses) > 0

        offset = 0
        for group_idx in range(self.n_groups):
            begin = offset
            if self.dict_num_candidates[group_idx] == 0:
                best_idx = -1
            else:
                end = offset + self.dict_num_candidates[group_idx]
                assert begin < end
                best_idx = np.argmin(losses[begin:end])
            local_best_indices.append(best_idx)
            offset = end
        return local_best_indices

    def get_total_instances(self):
        return sum(self.dict_num_candidates.values())


class SSDRAugment(DictAuxDataFeeder):
    def __init__(self, data, dictionary, ssdr_config, def_per_batch, max_def_length, data_info=None):
        self.max_def_length = max_def_length
        self.max_d_loc = ssdr_config.max_loc_length
        self.def_per_batch = def_per_batch
        self.raw_dictionary = dictionary
        self.use_ab_mapping_mask = ssdr_config.use_ab_mapping_mask
        tf_logging.debug("SSDRAugment init")
        tf_logging.debug("max_def_length: %d" % max_def_length)
        tf_logging.debug("max_d_loc: %d" % self.max_d_loc)
        tf_logging.debug("def_per_batch: %d" % self.def_per_batch)
        tf_logging.debug("use_ab_mapping_mask: {}".format(ssdr_config.use_ab_mapping_mask))
        super(SSDRAugment, self).__init__(data, data_info)

    def get_max_d_loc(self):
        return self.max_d_loc

    def dict_contains(self, word: str):
        return word in self.dict

    def encode_dict_as_feature(self, dictionary):
        new_dict = {}
        for word, entries in dictionary.items():
            enc_entries = []
            for dict_def in entries:
                d_input_ids = self.tokenizer.convert_tokens_to_ids(dict_def)
                d_input_ids = d_input_ids[:self.max_def_length]
                d_input_mask = [1] * len(d_input_ids)
                d_segment_ids = [0] * len(d_input_ids)

                d_input_ids = pad0(d_input_ids, self.max_def_length)
                d_input_mask = pad0(d_input_mask, self.max_def_length)
                d_segment_ids = pad0(d_segment_ids, self.max_def_length)

                e = d_input_ids, d_input_mask, d_segment_ids
                enc_entries.append(e)

            new_dict[word] = enc_entries
        return new_dict


    @staticmethod
    def drop_definitions(def_entries_list, def_per_batch):
        n_all_def = sum([len(entry) for entry in def_entries_list])

        iterations = 0

        inst_idx = 0
        while n_all_def >= def_per_batch:
            iterations += 1
            n_defs = len(def_entries_list[inst_idx])
            if n_defs > 1:
                drop_idx = random.randint(0, n_defs-1)
                def_entries_list[inst_idx] = def_entries_list[inst_idx][:drop_idx] \
                                                      + def_entries_list[inst_idx][drop_idx+1:]
                n_all_def -= 1

            inst_idx += 1
            if inst_idx == len(def_entries_list):
                inst_idx = 0

            assert iterations < 10000

    @staticmethod
    def _get_ab_mapping_mask(ab_map, a_size, b_size):
        ab_mapping_mask = np.zeros([a_size, b_size], int)

        for b_idx, a_idx in enumerate(ab_map):
            ab_mapping_mask[a_idx, b_idx] = 1

        return ab_mapping_mask

    @staticmethod
    def pack_data(def_entries_list, max_def_length, def_per_batch):
        ab_map = []
        batch_defs = []
        for idx, inst in enumerate(def_entries_list):
            for def_entry in inst:
                ab_map.append(idx)
                batch_defs.append(def_entry)

        assert len(ab_map) == len(batch_defs)
        while len(batch_defs) < def_per_batch:
            ab_map.append(0)
            dummy = [0] * max_def_length
            dummy_entry = (dummy, dummy, dummy)
            batch_defs.append(dummy_entry)

        return ab_map, batch_defs

    def get_random_data_indices(self, batch_size):
        indices = []
        for _ in range(batch_size):
            data_idx = random.randint(0, self.data_len - 1)
            indices.append(data_idx)
        return indices

    def get_random_batch(self, batch_size):
        problem_data = []
        def_entries_list = []
        for data_idx in self.get_random_data_indices(batch_size):
            appeared_words = self.data_info[data_idx]
            if appeared_words:
                word = pick1(appeared_words)
                def_entries_list.append(self.dict[word.word])
                d_location_ids = word.location
            else:
                def_entries_list.append([])
                d_location_ids = [0] * self.max_d_loc

            e = self.add_location(self.data[data_idx], d_location_ids)
            problem_data.append(e)

        self.drop_definitions(def_entries_list, self.def_per_batch)
        batch = self._get_batch(problem_data, def_entries_list, batch_size)
        return batch

    @staticmethod
    def add_location(data_entry, d_location_ids):
        input_ids, input_mask, segment_ids, y = data_entry
        e = input_ids, input_mask, segment_ids, d_location_ids, y
        return e

    def get_lookup_batch(self, batch_size):
        problem_data = []
        def_entries_list = []
        data_indices = self.get_random_data_indices(batch_size)
        for data_idx in data_indices:
            def_entries_list.append([])
            d_location_ids = [0] * self.max_d_loc
            e = self.add_location(self.data[data_idx], d_location_ids)
            problem_data.append(e)

        batch = self._get_batch(problem_data, def_entries_list, batch_size)
        return data_indices, batch

    def augment_dict_info(self, data_indices, ranked_locations):
        problem_data = []
        def_entries_list = []

        class NoWordException(Exception):
            pass
        for idx, data_idx in enumerate(data_indices):
            word = self.select_best_lookup_term(data_idx, ranked_locations[idx])
            if word is not None:
                def_entries_list.append(self.dict[word.word])
                d_location_ids = word.location
            else:
                def_entries_list.append([])
                d_location_ids = [0] * self.max_d_loc

            e = self.add_location(self.data[data_idx], d_location_ids)
            problem_data.append(e)

        batch_size = len(data_indices)
        self.drop_definitions(def_entries_list, self.def_per_batch)
        batch = self._get_batch(problem_data, def_entries_list, batch_size)
        return batch

    def get_all_batches(self, batch_size, f_return_indices=False):
        problem_data = []
        def_entries_list = []
        all_indice = []

        for data_idx in self.data_info.keys():
            all_indice.append(data_idx)
            appeared_words = self.data_info[data_idx]
            if appeared_words:
                word = pick1(appeared_words)
                def_entries_list.append(self.dict[word.word])
                d_location_ids = word.location
            else:
                def_entries_list.append([])
                d_location_ids = [0] * self.max_d_loc

            e = self.add_location(self.data[data_idx], d_location_ids)
            problem_data.append(e)

        n_insts = len(def_entries_list)
        assert n_insts == len(problem_data)

        batches = []
        for i in range(0, n_insts, batch_size):
            local_batch_len = min(batch_size, n_insts - i)
            if local_batch_len < batch_size:
                break
            current_problems = problem_data[i:i+local_batch_len]
            current_entries = def_entries_list[i:i+local_batch_len]
            self.drop_definitions(current_entries, self.def_per_batch)
            batch = self._get_batch(current_problems, current_entries, batch_size)
            if not f_return_indices:
                batches.append(batch)
            else:
                batches.append((all_indice[i:i+local_batch_len], batch))

        return batches


    def _pack_to_batches(self, all_instances, def_entries_list, batch_size):
        assert len(all_instances) == len(def_entries_list)
        idx = 0
        batches = []
        while idx < len(all_instances):
            local_def_entries_list = def_entries_list[idx:idx+batch_size]
            local_inst = all_instances[idx:idx + batch_size]
            while len(local_inst) < batch_size:
                local_inst.append(local_inst[0])
            self.drop_definitions(local_def_entries_list, self.def_per_batch)
            batch = self._get_batch(local_inst, local_def_entries_list, batch_size)
            batches.append(batch)
            idx += batch_size
        return batches

    # len(a_instances) <= batch_size
    # len(def_instances) <= self.def_per_batch
    def _get_batch(self, a_instances, def_instances, batch_size):
        assert len(a_instances) <= batch_size
        assert len(def_instances) <= self.def_per_batch
        ab_map, batch_defs = self.pack_data(def_instances, self.max_def_length, self.def_per_batch)
        ab_map = np.expand_dims(ab_map, 1)
        a_part = get_batches_ex(a_instances, batch_size, 5)[0]
        b_part = get_batches_ex(batch_defs, self.def_per_batch, 3)[0]

        if not self.use_ab_mapping_mask:
            batch = a_part + b_part + [ab_map]
        else:
            ab_mapping_mask = self._get_ab_mapping_mask(ab_map, batch_size, self.def_per_batch)
            batch = a_part + b_part + [ab_map] + [ab_mapping_mask]

        return batch

    def get_lookup_train_batches(self, batch_size, n_samples = 5):
        # generate instances with same main input, but different term for look up
        all_instances = []
        def_entries_list =[]
        info = LookupTrainInfo() # This object does not know about batches
        for idx in range(batch_size):
            data_idx = random.randint(0, self.data_len - 1)
            appeared_words = self.data_info[data_idx]

            group_instances = []
            n_samples = min(n_samples, len(appeared_words))
            candidate_words = random.sample(appeared_words, n_samples)
            for word in candidate_words:
                # get information about words
                d_location_ids = word.location
                def_entries_list.append(self.dict[word.word])
                e = self.add_location(self.data[data_idx], d_location_ids)
                all_instances.append(e)
                group_instances.append(e)
            info.add_group(group_instances)

        batches = self._pack_to_batches(all_instances, def_entries_list, batch_size)
        return batches, info

    def get_dummy_input_instance(self):
        data_idx = random.randint(0, self.data_len - 1)
        input_ids, input_mask, segment_ids, y = self.data[data_idx]
        d_location_ids = np.zeros([self.max_d_loc])
        return input_ids, input_mask, segment_ids, d_location_ids, y

    def get_lookup_training_batch(self, loss_list, batch_size, info: LookupTrainInfo):
        good_instances = info.select_best(loss_list)
        while len(good_instances) < batch_size:
            good_instances.append(self.get_dummy_input_instance())

        batch = self._get_batch(good_instances, [], batch_size)
        return batch
