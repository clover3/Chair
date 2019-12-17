import random
import sys
import time

from cache import *
from data_generator import tokenizer_wo_tf as tokenization
from misc_lib import TimeEstimator
from path import data_path
from sydney_manager import MarkedTaskManager
from tlm.wiki import bert_training_data as btd

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"

def parse_wiki(file_path):
    f = open(file_path, "r")
    documents = []
    doc = list()
    for line in f:
        if line.strip():
            doc.append(line)
        else:
            documents.append(doc)
            doc = list()
    return documents


class Worker:
    def __init__(self, out_path):
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

        self.masked_lm_prob = 0.15
        self.short_seq_prob = 0.1
        self.problem_per_job = 100 * 1000
        self.max_seq_length = 512
        self.max_predictions_per_seq = 20
        self.dupe_factor = 1
        self.out_dir = out_path

        seed = time.time()
        self.rng = random.Random(seed)
        print("Loading documents")
        self.documents = self.load_documents_from_pickle()
        print("Loading documents Done : ", len(self.documents))

    def load_documents_from_pickle(self):
        seg_id = self.rng.randint(0, 9)
        file_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/tokens/enwiki_train_tokens.{}"
        all_docs = []
        for j in range(100):
            full_id = seg_id * 100 + j
            f = open(file_path.format(full_id), "rb")
            all_docs.extend(pickle.load(f))
        return all_docs

    def load_documents(self):
        i = self.rng.randint(0,9)
        file_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train.txt.line.{}".format(i)
        print(file_path)
        docs = parse_wiki(file_path)

        out_docs = []
        # Empty lines are used as document delimiters
        ticker = TimeEstimator(len(docs))
        for doc in docs:
            out_docs.append([])
            for line in doc:
                line = line.strip()
                tokens = self.tokenizer.tokenize(line)
                if tokens:
                    out_docs[-1].append(tokens)


            ticker.tick()
        assert out_docs[3]
        return out_docs


    def work(self, job_id):
        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        instances = btd.create_training_instances(
            self.documents, self.tokenizer, self.max_seq_length, self.dupe_factor,
            self.short_seq_prob, self.masked_lm_prob, self.max_predictions_per_seq,
            self.rng)
        btd.write_instance_to_example_files(instances, self.tokenizer, self.max_seq_length,
                                        self.max_predictions_per_seq, [output_file])


def main():
    mark_path = os.path.join(working_path, "wiki_p2_mark")
    out_path = os.path.join(working_path, "tf")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    mtm = MarkedTaskManager(100, mark_path, 1)
    worker = Worker(out_path)

    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)

def simple():
    out_path = os.path.join(working_path, "tf")
    worker = Worker(out_path)

    worker.work(int(sys.argv[1]))


if __name__ == "__main__":
    main()

