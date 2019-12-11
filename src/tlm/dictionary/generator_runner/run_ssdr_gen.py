import tlm.data_gen.lm_worker
from cache import load_from_pickle
from tlm.dictionary import data_gen

if __name__ == "__main__":
    d = load_from_pickle("webster_parsed_w_cls")
    batch_size = 32
    def_per_batch = 320

    def get_generator():
        return data_gen.SenseSelectingDictionaryReaderGen(d, batch_size, def_per_batch, 96)
    runner = tlm.data_gen.lm_worker.LMJobRunner(4000, "ssdr2", get_generator)
    runner.start()

