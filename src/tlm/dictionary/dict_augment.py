from cache import load_cache, save_to_pickle, load_from_pickle
from tf_util.tf_logging import tf_logging
from tlm.dictionary.data_gen import SSDRAugment, DictAugment
from tlm.model_cnfig import JsonConfig
from tlm.training.train_flags import FLAGS


NAME_WSSDR = "wssdr"
NAME_dict_1 = "dict_1"


class DictAugmentedDataLoader:
    def __init__(self, feeder_name, data_loader):
        self.data_loader = data_loader
        if feeder_name == NAME_dict_1:
            self.feeder_getter = self.get_dict1_feeder
        elif feeder_name == NAME_WSSDR:
            self.feeder_getter = self.get_wssdr_feeder
        else:
            raise Exception()

        self.feeder_name = feeder_name


    def _load_data_feeder(self, split_name, original_data_getter):
        original_data = original_data_getter()
        cache_name = split_name + "_feeder." + self.feeder_name
        data_feeder = load_cache(cache_name)
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
    def get_dict1_feeder(data):
        dictionary = load_from_pickle("webster")
        return DictAugment(data, dictionary)

    def get_train_feeder(self):
        return self._load_data_feeder("train_data", self.data_loader.get_train_data)

    def get_dev_feeder(self):
        return self._load_data_feeder("dev_data", self.data_loader.get_dev_data)
