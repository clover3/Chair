import os
import pickle

from data_generator import job_runner
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.data_gen.base import UnmaskedPairedDataGen


def get_all_file_name_list():
    f = open("/mnt/nfs/work3/youngwookim/data/clueweb12-B13-dir_list.txt", "r")
    dropping_tail = ".warc.gz"
    l = []
    for line in f:
        name = os.path.basename(line.strip())
        name = name[:-len(dropping_tail)]
        l.append(name)
    return l


def filter_short(doc_dict):
    for d in doc_dict:
        if sum([len(t) for t in d]) < 120:
            pass
        else:
            yield d

class CluewebLMWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, generator):
        self.out_dir = out_path
        self.generator = generator
        self.token_path = "/mnt/nfs/work3/youngwookim/data/clueweb12-B13_tokens/"
        self.all_name = get_all_file_name_list()

    def load_tokens_by_job_id(self, job_id):
        d = {}
        st = job_id * 10
        ed = (job_id+1) * 10
        for file_name in self.all_name[st:ed]:
            path = os.path.join(self.token_path, file_name)
            try:
                data : dict= pickle.load(open(path, "rb"))
                d.update(data)
            except FileNotFoundError as e:
                print(e)
        print("Loaded {} docs for {}".format(len(d), job_id))
        return d

    def work(self, job_id):
        token_d = self.load_tokens_by_job_id(job_id)
        docs = token_d.values()
        docs = filter_short(docs)
        insts = self.generator.create_instances_from_documents(docs)
        output_file = os.path.join(self.out_dir, str(job_id))
        self.generator.write_instances(insts, output_file)


if __name__ == "__main__":
    generator = UnmaskedPairedDataGen()
    JobRunner(sydney_working_dir, 1208, "clueweb12_13B_no_short", lambda x: CluewebLMWorker(x, generator)).start()


