import os

from crs.contradiction_pair.pair_payload import PairEncoder
from data_generator import job_runner
from data_generator.argmining import ukp
from data_generator.job_runner import sydney_working_dir, JobRunner
from misc_lib import exist_or_mkdir
from tlm.ukp.sydney_data import ukp_load_tokens_for_topic, sydney_get_ukp_ranked_list, non_cont_load_tokens_for_topic, \
    sydney_get_nc_ranked_list

non_cont_topics = ["hydroponics", "weather", "restaurant", "wildlife_extinction", "james_allan"]
n_topic2 = 3

class PairPayloadWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, top_k, generator, drop_head=0):
        self.n_repeat = 30
        self.out_dir = out_path
        self.generator = generator
        self.drop_head = drop_head
        self.top_k = top_k

    def get_cont_docs(self, topic):
        ranked_list = sydney_get_ukp_ranked_list()[topic]
        all_tokens = ukp_load_tokens_for_topic(topic)

        return self.select(all_tokens, ranked_list)

    def select(self, all_tokens, ranked_list):
        print("Ranked list contains {} docs, selecting top-{}".format(len(ranked_list), self.top_k))
        if self.drop_head:
            print("Drop first {}".format(self.drop_head))
        doc_ids = [doc_id for doc_id, _, _ in ranked_list[self.drop_head:self.top_k]]
        docs = [all_tokens[doc_id] for doc_id in doc_ids if doc_id in all_tokens]
        return docs

    def get_non_cont_docs(self, topic):
        ranked_list = sydney_get_nc_ranked_list()[topic]
        all_tokens = non_cont_load_tokens_for_topic(topic)
        return self.select(all_tokens, ranked_list)

    def work(self, job_id):
        topic_selector = int(job_id / self.n_repeat)
        topic1_idx = int(topic_selector / n_topic2)
        topic2_idx = topic_selector % n_topic2
        topic1 = ukp.all_topics[topic1_idx]
        topic2 = non_cont_topics[topic2_idx]
        docs = self.get_cont_docs(topic1)
        docs += self.get_non_cont_docs(topic2)
        insts = self.generator.create_instances_from_documents(docs)

        sub_dir_name = "{}_{}".format(topic1, topic2)
        out_dir = os.path.join(self.out_dir, sub_dir_name)
        exist_or_mkdir(out_dir)
        output_file = os.path.join(out_dir, str(job_id))
        self.generator.write_instances(insts, output_file)


if __name__ == "__main__":
    top_k = 1000
    generator = PairEncoder(number_of_pairs=10000)
    num_jobs = 30 * 2
    JobRunner(sydney_working_dir, num_jobs, "pair_payload_two_topic",
              lambda x: PairPayloadWorker(x, top_k, generator)).start()


