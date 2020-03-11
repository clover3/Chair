


#
# Format : A_seg =[ max_sent_len * num_sent]
# Format : B_seg =[ max_seq_len - max_sent_len]
#
#
#
import os
import pickle

import data_generator.argmining.ukp_header
from data_generator import job_runner
from data_generator.job_runner import JobRunner, sydney_working_dir
from misc_lib import flatten, lmap, get_dir_files
from tf_util.record_writer_wrap import RecordWriterWrap
from tf_util.tf_logging import tf_logging
from tlm.data_gen.base import LMTrainBase, format_tokens_pair_n_segid, truncate_seq_pair, get_basic_input_feature, \
    log_print_feature
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.ukp.sydney_data import sydney_get_ukp_ranked_list


class CarryOverGen(LMTrainBase):
    def __init__(self):
        super(CarryOverGen, self).__init__()
        self.max_sent_length = 50
        self.max_b_seg_length = self.max_seq_length - self.max_sent_length - 3

    def create_instances_from_document(self, document):
        # check document length to determine if the document is appropriate to do
        if len(document) < 5 :
            return
        if sum(lmap(len, document)) < self.max_seq_length * 2:
            return

        doc_len = len(document)
        # iterate over collection by modifying the windows
        payload = []
        for st in range(min(doc_len, 100)):
            cur_chunks = []

            chunk_len = 0
            idx = st
            while idx < len(document) and chunk_len < self.max_b_seg_length:
                chunk_len += len(document[idx])
                cur_chunks.append(document[idx])
                idx += 1
            ed = idx
            tokens = flatten(cur_chunks)
            tokens = tokens[:self.max_b_seg_length]

            e = (tokens, [], -1, st, ed)
            payload.append(e)
            for i in range(min(doc_len, 100)):
                if i < st or ed <= i:
                    sent_a = document[i][:self.max_sent_length]

                    e = (tokens, sent_a, i, st, ed)
                    payload.append(e)

            if len(payload) % 10000 == 1:
                print("num payload : ", len(payload))

            if len(payload) > 1000:
                break

        for tokens, sent_a, a_loc, st, ed in payload:
            truncate_seq_pair(tokens, sent_a, self.max_seq_length -3, self.rng)
            tokens, segment_ids = format_tokens_pair_n_segid(tokens, sent_a)
            yield tokens, segment_ids, a_loc, st, ed

    def write_instances(self, new_inst_list, outfile):
        writer = RecordWriterWrap(outfile)
        example_numbers = []

        for (inst_index, instance) in enumerate(new_inst_list):
            tokens, segment_ids, inst_id = instance
            features = get_basic_input_feature(self.tokenizer,
                                               self.max_seq_length,
                                               tokens,
                                               segment_ids)
            features["instance_id"] = create_int_feature([inst_id])

            writer.write_feature(features)
            if inst_index < 20:
                log_print_feature(features)
        writer.close()

        tf_logging.info("Wrote %d total instances", writer.total_written)

        return example_numbers

class CarryOverAnalyze(job_runner.WorkerInterface):
    def __init__(self, out_path, top_k):
        self.out_dir = out_path
        self.top_k = top_k
        self.token_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_tokens/"
        self.max_inst_per_job = 1000 * 1000
        self.inst_cnt = 0

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

    def get_payload_id(self, job_id):
        payload_id = self.max_inst_per_job * job_id + self.inst_cnt
        self.inst_cnt += 1
        assert self.inst_cnt < self.max_inst_per_job
        return payload_id

    def work(self, job_id):
        topic = data_generator.argmining.ukp_header.all_topics[job_id]
        ranked_list = sydney_get_ukp_ranked_list()[topic]
        print("Ranked list contains {} docs, selecting top-{}".format(len(ranked_list), self.top_k))
        doc_ids = [doc_id for doc_id, _, _ in ranked_list[:self.top_k]]
        all_tokens = self.load_tokens_for_topic(topic)

        generator = CarryOverGen()
        payload_info = []
        modified_insts = []
        for doc_id in doc_ids[:200]:
            doc = all_tokens[doc_id]
            insts = generator.create_instances_from_document(doc)
            for tokens, segment_ids, sent_loc, st, ed in insts:
                payload_id = self.get_payload_id(job_id)
                payload_info.append({
                    'payload_id':payload_id,
                    'doc_id':doc_id,
                    'hint_loc':sent_loc,
                    'st':st,
                    'ed':ed,
                })
                modified_insts.append((tokens, segment_ids, payload_id))

        output_file = os.path.join(self.out_dir, topic.replace(" ", "_"))
        generator.write_instances(modified_insts, output_file)
        info_output_file = output_file + ".info"
        pickle.dump(payload_info, open(info_output_file, "wb"))


if __name__ == "__main__":
    top_k = 1000
    num_jobs = len(data_generator.argmining.ukp_header.all_topics) - 1
    JobRunner(sydney_working_dir, num_jobs, "carry_over_1", lambda x: CarryOverAnalyze(x, top_k)).start()


