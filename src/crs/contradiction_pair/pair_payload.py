import os
import random
import time

import data_generator.tokenizer_wo_tf as tokenization
from cpath import data_path
from data_generator import job_runner
from data_generator.argmining import ukp
from data_generator.job_runner import JobRunner, sydney_working_dir
from misc_lib import flatten, pick1
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import truncate_seq_pair, format_tokens_pair_n_segid, SegmentInstance, \
    get_basic_input_feature, log_print_inst
from tlm.ukp.sydney_data import ukp_load_tokens_for_topic, sydney_get_ukp_ranked_list


class PairEncoder:
    def __init__(self, number_of_pairs = 10000):
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

        self.number_of_pairs = number_of_pairs
        self.max_seq_length = 200
        self.rng = random.Random(time.time())

    def create_instances_from_documents(self, docs):
        sents = flatten(docs)
        def pick_short_sent():
            tokens = pick1(sents)
            while len(tokens) > 100:
                tokens = pick1(sents)
            return tokens


        instances = []
        for _ in range(self.number_of_pairs):
            tokens_a = pick_short_sent()
            tokens_b = pick_short_sent()
            target_seq_length = self.max_seq_length - 3

            truncate_seq_pair(tokens_a, tokens_b, target_seq_length, self.rng)

            tokens, segment_ids = format_tokens_pair_n_segid(tokens_a, tokens_b)
            instance = SegmentInstance(
                tokens=tokens,
                segment_ids=segment_ids)
            instances.append(instance)

        return instances

    def write_instances(self, new_inst_list, outfile):
        writer = RecordWriterWrap(outfile)
        example_numbers = []

        for (inst_index, instance) in enumerate(new_inst_list):
            features = get_basic_input_feature(self.tokenizer,
                                               self.max_seq_length,
                                               instance.tokens,
                                               instance.segment_ids)

            writer.write_feature(features)
            if inst_index < 20:
                log_print_inst(instance, features)
        writer.close()

        return example_numbers



class PairPayloadWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, top_k, generator, drop_head=0):
        self.out_dir = out_path
        self.generator = generator
        self.drop_head = drop_head
        self.top_k = top_k
        self.token_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_tokens/"


    def work(self, job_id):
        topic = ukp.all_topics[0]
        ranked_list = sydney_get_ukp_ranked_list()[topic]
        print("Ranked list contains {} docs, selecting top-{}".format(len(ranked_list), self.top_k))
        if self.drop_head:
            print("Drop first {}".format(self.drop_head))
        doc_ids = [doc_id for doc_id, _, _ in ranked_list[self.drop_head:self.top_k]]
        all_tokens = ukp_load_tokens_for_topic(topic)

        docs = [all_tokens[doc_id] for doc_id in doc_ids if doc_id in all_tokens]
        insts = self.generator.create_instances_from_documents(docs)
        output_file = os.path.join(self.out_dir, str(job_id))
        self.generator.write_instances(insts, output_file)


if __name__ == "__main__":
    top_k = 1000
    num_jobs = 30
    ##
    generator = PairEncoder(number_of_pairs = 10000)
    JobRunner(sydney_working_dir, num_jobs, "pair_payload", lambda x: PairPayloadWorker(x, top_k, generator)).start()


