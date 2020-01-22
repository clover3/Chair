import os
import pickle

from data_generator import job_runner
from data_generator.argmining import ukp
from data_generator.job_runner import JobRunner, sydney_working_dir
from misc_lib import get_dir_files
from tlm.data_gen.base import UnmaskedPairedDataGen
from tlm.ukp.load_multiple_ranked_list import sydney_get_ukp_ranked_list


class UkpWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, top_k):
        self.out_dir = out_path
        self.top_k = top_k
        self.token_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_tokens/"

    def load_tokens_for_topic(self, topic):
        d = {}
        for path in get_dir_files(self.token_path):
            if topic.replace(" ", "_") in path:
                d.update(pickle.load(open(path, "rb")))
        print("Loaded {} docs for {}".format(len(d), topic))

        for idx, doc_id in enumerate(d.keys()):
            print(doc_id, end=", ")
            if idx > 3:
                print()
                break
        return d

    def work(self, job_id):
        topic = ukp.all_topics[job_id]
        ranked_list = sydney_get_ukp_ranked_list()[topic]
        print("Ranked list contains {} docs, selecting top-{}".format(len(ranked_list), self.top_k))
        doc_ids = [doc_id for doc_id, _, _ in ranked_list[:self.top_k]]
        all_tokens = self.load_tokens_for_topic(topic)

        docs = [all_tokens[doc_id] for doc_id in doc_ids if doc_id in all_tokens]
        generator = UnmaskedPairedDataGen()
        insts = generator.create_instances_from_documents(docs)
        output_file = os.path.join(self.out_dir, topic.replace(" ", "_"))
        generator.write_instances(insts, output_file)


if __name__ == "__main__":
    top_k = 1000
    num_jobs = len(ukp.all_topics) - 1
    JobRunner(sydney_working_dir, num_jobs, "ukp_1_debug", lambda x:UkpWorker(x, top_k)).start()


