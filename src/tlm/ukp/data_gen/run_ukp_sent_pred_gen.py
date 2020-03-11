import os

import data_generator.argmining.ukp_header
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.ukp.data_gen.run_ukp_gen1 import UkpWorker
from tlm.ukp.data_gen.ukp_pred_gen import UkpSentGen
from tlm.ukp.sydney_data import sydney_get_ukp_ranked_list


class UkpSentPredWorker(UkpWorker):
    def work(self, job_id):
        topic = data_generator.argmining.ukp_header.all_topics[job_id]
        ranked_list = sydney_get_ukp_ranked_list()[topic]
        print("Ranked list contains {} docs, selecting top-{}".format(len(ranked_list), self.top_k))
        if self.drop_head:
            print("Drop first {}".format(self.drop_head))
        doc_ids = [doc_id for doc_id, _, _ in ranked_list[self.drop_head:self.top_k]]
        all_tokens = self.load_tokens_for_topic(topic)

        docs = [all_tokens[doc_id] for doc_id in doc_ids if doc_id in all_tokens]
        insts = self.generator.create_instances_from_documents(topic, docs)
        output_file = os.path.join(self.out_dir, topic.replace(" ", "_"))
        self.generator.write_instances(insts, output_file)


if __name__ == "__main__":
    top_k = 150000
    num_jobs = len(data_generator.argmining.ukp_header.all_topics) - 1

    generator = UkpSentGen()
    generator.max_seq_length = 128
    JobRunner(sydney_working_dir, num_jobs, "ukp_sent_pred", lambda x: UkpSentPredWorker(x, top_k, generator)).start()


