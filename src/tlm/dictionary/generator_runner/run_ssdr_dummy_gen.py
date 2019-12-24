import tlm.data_gen.lm_worker
from cache import load_from_pickle
from tlm.dictionary import data_gen

if __name__ == "__main__":
    d = load_from_pickle("webster_parsed_w_cls")

    new_d = {}
    for word, def_list in d.items():
        new_def = []
        for _ in def_list:
            new_def.append([])
        new_d[word] = new_def

    batch_size = 32
    def_per_batch = 320

    def get_generator():
        return data_gen.SenseSelectingDictionaryReaderGen(new_d, batch_size, def_per_batch, 96, True)
    runner = tlm.data_gen.lm_worker.LMJobRunner(1000, "ssdr_dummy3", get_generator)
    runner.start()

