from dictionary.reader import DictionaryReader
from tlm.dictionary.data_gen import dictionary_encoder
from data_generator.common import get_tokenizer
from cache import *
from sydney_manager import MarkedTaskManager
from tlm.dictionary.data_gen import DictTrainGen, Dictionary
from tlm.data_gen import run_unmasked_pair_gen
from tlm.tf_logging import tf_logging
import logging
from misc_lib import exist_or_mkdir

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"


def encode_dictionary():
    path = "c:\\work\\Data\\webster\\webster_headerless.txt"
    d1 = DictionaryReader.open(path)
    d = dictionary_encoder(d1.entries, get_tokenizer())
    save_to_pickle(d, "webster")


class DGenWorker(run_unmasked_pair_gen.Worker):
    def __init__(self, out_path):
        super(DGenWorker, self).__init__(out_path)
        self.out_dir = out_path
        exist_or_mkdir(out_path)
        d = Dictionary(load_from_pickle("webster"))
        self.gen = DictTrainGen(d)


def main():
    mark_path = os.path.join(working_path, "dict_reader3_mark")
    out_path = os.path.join(working_path, "dict_reader3")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    mtm = MarkedTaskManager(4000, mark_path, 1)
    worker = DGenWorker(out_path)
    worker.gen.f_hide_word = False

    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)


def simple():
    tf_logging.setLevel(logging.INFO)
    out_path = os.path.join(working_path, "dict_reader3")
    exist_or_mkdir(out_path)
    worker = DGenWorker(out_path)
    worker.gen.f_hide_word = False
    worker.work(1)


def generate_test():
    tf_logging.setLevel(logging.INFO)
    out_path = os.path.join(working_path, "dict_reader2_test_dict")
    worker = DGenWorker(out_path)
    worker.gen.drop_none_dict = True
    worker.work(1)

    out_path = os.path.join(working_path, "dict_reader2_test_no_dict")
    worker = DGenWorker(out_path)
    worker.gen.no_dict_assist = True
    worker.work(1)


if __name__ == "__main__":
    main()


