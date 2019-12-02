from cache import load_from_pickle
from tlm.dictionary import data_gen
from tlm.dictionary.generator_runner import wiki_dict_job_runner

if __name__ == "__main__":
    d = load_from_pickle("webster_parsed_w_cls")
    batch_size = 32
    def_per_batch = 320
    def get_generator():
        return data_gen.SenseSelectingDictionaryReaderGen(d, batch_size, def_per_batch, 96)
    runner = wiki_dict_job_runner.PDictJobRunner(4000, "ssdr", get_generator)
    runner.start()

