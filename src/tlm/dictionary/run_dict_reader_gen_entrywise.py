from data_generator import job_runner
from tlm.dictionary import wiki_dict_job_runner
from tlm.dictionary import data_gen
from cache import load_from_pickle
from functools import partial

if __name__ == "__main__":
    class EntryWiseGen(data_gen.DictTrainGen):
        def __init__(self):
            d = load_from_pickle("webster_parsed_w_cls")
            dictionary = data_gen.RandomEntryDictionary(d)
            super(EntryWiseGen, self).__init__(dictionary)
            self.drop_short_word = False
            self.max_def_length = 128

    runner = wiki_dict_job_runner.PDictJobRunner(1000, "dict_entrywise", EntryWiseGen)
    runner.start()

