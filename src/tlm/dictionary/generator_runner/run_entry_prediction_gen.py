import logging
import random
import sys

from cache import *
from data_generator.common import get_tokenizer
from dictionary.reader import DictionaryReader, DictionaryParser, all_pos_list
from misc_lib import TimeEstimator, lmap
from misc_lib import exist_or_mkdir
from sydney_manager import MarkedTaskManager
from tlm.dictionary.data_gen import DictEntryPredictGen
from tlm.tf_logging import tf_logging

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"

# Special Tokens
#
# Generic Tokens
# 0: PAD_ID =
# 1: EOS_ID = [unused0]
# 2: CLS_ID = [unused1]
# 3: SEP_ID = [unused2]
# 4: OOV_ID = [unused3]
#
# Dictionary Specific tokens
# TOKEN_LINE_SEP = "[unused5]"
# TOKEN_DEF_SEP = "[unused6]"
# POS related tokens
# [unused10] ~~ [unused50]


def reserve_pos_as_special_tokens():
    assert len(all_pos_list) < 40
    template = "[unused{}]"
    start_idx = 10
    pos_to_token = {}
    for i, pos in enumerate(all_pos_list):
        pos_to_token[pos] = template.format(i+start_idx)

    return pos_to_token


def encode_parsed_dictionary():
    tokenizer = get_tokenizer()
    path = "c:\\work\\Data\\webster\\webster_headerless.txt"
    d1 = DictionaryReader.open(path)
    d2 = DictionaryParser(d1)

    out_lines = 0
    pos_to_token = reserve_pos_as_special_tokens()

    def encode_pos(pos_list):
        output = []
        for pos in pos_list:
            output.append(pos_to_token[pos])
        return output

    result_dict = {}
    ticker = TimeEstimator(len(d2.word2entry))

    for word, word_entry in d2.word2entry.items():
        if out_lines < 1000:
            print(word)

        result_dict[word] = []
        for sub_entry in word_entry.sub_entries:
            for definition in sub_entry.enum_definitions():
                def_str = definition
                pos_str = sub_entry.get_pos_as_str()

                out_str = pos_str + " " + def_str
                pos_tokens = encode_pos(sub_entry.get_pos_list())
                def_tokens = tokenizer.tokenize(def_str)

                out_tokens = pos_tokens + def_tokens
                result_dict[word].append(out_tokens)
                if out_lines < 1000:
                    print(out_str)
                    out_lines += 1
        ticker.tick()

    save_to_pickle(result_dict, "webster_parsed")


def add_cls_to_parsed():
    d = load_from_pickle("webster_parsed")

    def add_cls(def_tokens):
        return ["[CLS]"] + def_tokens

    new_d = {}
    for word, def_list in d.items():
        new_def_list = lmap(add_cls, def_list)
        new_d[word.lower()] = new_def_list

    save_to_pickle(new_d, "webster_parsed_w_cls")


class Worker:
    def __init__(self, example_out_path, n_out_path):
        self.example_out_dir = example_out_path
        self.n_out_path = n_out_path

        d = load_from_pickle("webster_parsed")

        d = {k.lower():v for k,v in d.items()}
        max_def_entry = 10
        self.gen = DictEntryPredictGen(d, max_def_entry)

    def work(self, job_id):
        doc_id = job_id
        if doc_id >= 1000:
            doc_id = doc_id % 1000

        docs = self.gen.load_doc_seg(doc_id)
        example_file = os.path.join(self.example_out_dir, "{}".format(job_id))
        insts = self.gen.create_instances_from_documents(docs)
        random.shuffle(insts)
        n_list = self.gen.write_instances(insts, example_file)

        n_out_path = os.path.join(self.n_out_path, "{}".format(job_id))
        f = open(n_out_path, "w")
        for n in n_list:
            f.write("{}\n".format(n))
        f.close()


def init_worker():
    out_path1 = os.path.join(working_path, "entry_prediction_tf")
    out_path2 = os.path.join(working_path, "entry_prediction_n")
    exist_or_mkdir(out_path1)
    exist_or_mkdir(out_path2)

    worker = Worker(out_path1, out_path2)
    return worker


def main():
    mark_path = os.path.join(working_path, "entry_prediction_mark")
    mtm = MarkedTaskManager(1000, mark_path, 1)

    worker = init_worker()
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)
    ##

def simple():
    tf_logging.setLevel(logging.INFO)
    worker = init_worker()
    worker.work(int(sys.argv[1]))


if __name__ == "__main__":
    add_cls_to_parsed()

