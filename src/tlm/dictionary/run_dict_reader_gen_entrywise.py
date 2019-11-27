
from data_generator import job_runner
from tlm.dictionary import wiki_dict_job_runner
from tlm.dictionary import data_gen
from cache import load_cache
from functools import partial

if __name__ == "__main__":
    class EntryWiseGen(data_gen.DictTrainGen):
        def __init__(self):
            d = load_cache("entry_encoded_dict")
            super(EntryWiseGen, self).__init__(d)

    runner = wiki_dict_job_runner.PDictJobRunner(1000, "dict_entrywise", EntryWiseGen)
    runner.start()

