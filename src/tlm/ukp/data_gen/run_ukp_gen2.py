import os
import pickle
import random

import data_generator.argmining.ukp_header
from data_generator import job_runner
from data_generator.job_runner import JobRunner, sydney_working_dir
from misc_lib import get_dir_files
from tlm.data_gen.lm_datagen import UnmaskedPairedDataGen
from tlm.ukp.sydney_data import sydney_get_ukp_ranked_list


def load_tokens_for_topic(topic):
    d = {}
    token_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_tokens/"
    for path in get_dir_files(token_path):
        if topic.replace(" ", "_") in path:
            data = pickle.load(open(path, "rb"))
            if len(data) < 10000:
                print("{} has {} data".format(path, len(data)))
            d.update(data)
    print("Loaded {} docs for {}".format(len(d), topic))

    return d


class UkpWorker2(job_runner.WorkerInterface):
    def __init__(self, out_path, generator, top_k):
        self.out_dir = out_path
        self.generator = generator
        self.top_k = top_k

    def work(self, job_id):
        topic = data_generator.argmining.ukp_header.all_topics[job_id]
        ranked_list = sydney_get_ukp_ranked_list()[topic]
        all_doc_ids = list([doc_id for doc_id, _, _ in ranked_list])
        print("Ranked list contains {} docs, selecting top-{}".format(len(ranked_list), self.top_k))
        all_doc_ids = all_doc_ids[:self.top_k]

        random.shuffle(all_doc_ids)
        split = int(len(all_doc_ids) * 0.9)

        train_doc_ids = all_doc_ids[:split]
        val_doc_ids = all_doc_ids[split:]
        all_tokens = load_tokens_for_topic(topic)

        for doc_ids, split_name in [(train_doc_ids, "train"), (val_doc_ids, "val")]:
            docs = [all_tokens[doc_id] for doc_id in doc_ids if doc_id in all_tokens]
            insts = self.generator.create_instances_from_documents(docs)
            output_file = os.path.join(self.out_dir, topic.replace(" ", "_")) + "_" + split_name
            self.generator.write_instances(insts, output_file)


if __name__ == "__main__":
    num_jobs = len(data_generator.argmining.ukp_header.all_topics) - 1
    generator = UnmaskedPairedDataGen()
    top_k = 1000000
    JobRunner(sydney_working_dir, num_jobs, "ukp_10000_all", lambda x: UkpWorker2(x, generator, top_k)).start()


