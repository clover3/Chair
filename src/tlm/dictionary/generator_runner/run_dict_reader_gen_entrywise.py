from cache import load_from_pickle
from tlm.dictionary import data_gen
from tlm.dictionary.generator_runner import wiki_dict_job_runner

if __name__ == "__main__":
    class EntryWiseGen(data_gen.DictTrainGen):
        def __init__(self):
            d = load_from_pickle("webster_parsed_w_cls")
            dictionary = data_gen.RandomEntryDictionary(d)
            super(EntryWiseGen, self).__init__(dictionary, 128)
            self.drop_short_word = False

    runner = wiki_dict_job_runner.PDictJobRunner(4000, "dict_entrywise_128", EntryWiseGen)
    runner.start()

