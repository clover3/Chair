import os
import pickle

import data_generator.argmining.ukp_header
from data_generator import job_runner
from data_generator.job_runner import JobRunner, sydney_working_dir
from misc_lib import get_dir_files
from tlm.data_gen.lm_datagen import UnmaskedPairedDataGen
from tlm.ukp.sydney_data import sydney_get_ukp_ranked_list


class UkpWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, top_k, generator, drop_head=0):
        self.out_dir = out_path
        self.generator = generator
        self.drop_head = drop_head
        self.top_k = top_k
        self.token_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_tokens/"

    def load_tokens_for_topic(self, topic):
        d = {}
        for path in get_dir_files(self.token_path):
            if topic.replace(" ", "_") in path:
                data = pickle.load(open(path, "rb"))
                if len(data) < 10000:
                    print("{} has {} data".format(path, len(data)))
                d.update(data)
        print("Loaded {} docs for {}".format(len(d), topic))

        return d

    def work(self, job_id):
        topic = data_generator.argmining.ukp_header.all_topics[job_id]
        ranked_list = sydney_get_ukp_ranked_list()[topic]
        print("Ranked list contains {} docs, selecting top-{}".format(len(ranked_list), self.top_k))
        if self.drop_head:
            print("Drop first {}".format(self.drop_head))
        doc_ids = [doc_id for doc_id, _, _ in ranked_list[self.drop_head:self.top_k]]
        all_tokens = self.load_tokens_for_topic(topic)

        docs = [all_tokens[doc_id] for doc_id in doc_ids if doc_id in all_tokens]
        insts = self.generator.create_instances_from_documents(docs)
        output_file = os.path.join(self.out_dir, topic.replace(" ", "_"))
        self.generator.write_instances(insts, output_file)


if __name__ == "__main__":
    drop_head = 10000
    top_k = 150000
    num_jobs = len(data_generator.argmining.ukp_header.all_topics) - 1
    generator = UnmaskedPairedDataGen()
    JobRunner(sydney_working_dir, num_jobs, "ukp_150K_wo_10K", lambda x: UkpWorker(x, top_k, generator, drop_head)).start()


