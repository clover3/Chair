
# My library
import json
import os
import pickle
import random
### Python Library
import sys
from collections import Counter

import cpath
from data_generator import tokenizer_b as tokenization
from galagos.basic import load_galago_ranked_list
from list_lib import left
from misc_lib import TimeEstimator, tprint, CodeTiming
from models.classic.stopword import load_stopwords
from sydney_manager import MarkedTaskManager, ReadyMarkTaskManager
from tlm.feature2 import FeatureExtractor
from tlm.feature_extractor import libsvm_str
from tlm.retrieve_lm import per_doc_posting_server
from tlm.retrieve_lm.galago_query_maker import clean_query
from tlm.retrieve_lm.retreive_candidates import translate_mask2token_level, remove
from tlm.retrieve_lm.sample_segments import get_doc_sent, extend
from tlm.retrieve_lm.segment2problem import generate_mask
from tlm.retrieve_lm.select_sentence import get_random_sent
from tlm.retrieve_lm.stem import CacheStemmer, stemmed_counter
from tlm.retrieve_lm.tf_instance_writer import TFRecordMaker
from tlm.retrieve_lm.tf_record_writer import filter_instances
from tlm.two_seg_pretraining import write_predict_instance


class Pipeline:
    def __init__(self):
        tprint("Pipeline Init")
        self.stemmer = CacheStemmer()
        vocab_file = os.path.join(cpath.data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.iteration_dir = "/mnt/scratch/youngwookim/data/tlm_iter1"
        if not os.path.exists("/mnt/scratch/youngwookim/"):
            self.iteration_dir = "/mnt/nfs/work3/youngwookim/data/tlm_iter1"
        self.seg_max_seq = 256
        self.model_max_seq = 512
        self.rng = random.Random(0)
        self.masked_lm_prob = 0.15
        self.short_seq_prob = 0.1
        self.inst_per_job = 1000
        self.stopword = load_stopwords()
        self.pr = FeatureExtractor(self.seg_max_seq-3)
        self.tf_record_maker = None
        self.code_tick = CodeTiming()
        tprint("Pipeline Init Done")

    def run_A(self, job_id):
        output = []
        queries = []
        ticker = TimeEstimator(self.inst_per_job)
        for i in range(self.inst_per_job):
            p, q = self.process_A()
            qid = job_id * self.inst_per_job + i

            output.append((p, qid))
            queries.append((qid, q))
            ticker.tick()

        self.save_query(job_id, queries)
        self.save_output_A(job_id, output)

    def run_B(self, job_id):
        if self.pr.doc_posting is None:
            self.pr.doc_posting = per_doc_posting_server.load_dict()

        output_A = self.load_output_A(job_id)
        candi_docs = self.load_candidate_docs(job_id)
        feature_str_list = []
        seg_candi_list = []
        ticker = TimeEstimator(self.inst_per_job)
        for i in range(self.inst_per_job):
            problem, qid = output_A[i]
            qid_str = str(qid)
            if qid_str in candi_docs:
                doc_candi = candi_docs[qid_str]
                seg_candi, features = self.process_B(problem, doc_candi)
                fstr = "\n".join([libsvm_str(qid, 0, f) for f in features])
                feature_str_list.append(fstr)
                seg_candi_list.append(seg_candi)
            else:
                feature_str_list.append([])
                seg_candi_list.append([])
            ticker.tick()

            if i % 100 == 3:
                self.code_tick.print()


        self.save("seg_candi_list", job_id, seg_candi_list)
        self.save_ltr(job_id, feature_str_list)

    def run_C(self, job_id):
        if self.tf_record_maker is None:
            self.tf_record_maker = TFRecordMaker(self.model_max_seq)


        output_A = self.load_output_A(job_id)
        seg_candi_list = self.load("seg_candi_list",job_id)
        ltr_result = self.load_ltr(job_id)

        uid_list = []
        tf_record_list = []
        for i in range(self.inst_per_job):
            problem, qid = output_A[i]
            insts_id = "{}_{}".format(job_id, i)
            r = self.process_C(problem, seg_candi_list[i], ltr_result[i])
            for idx, tf_record in r:
                tf_id = insts_id + "_{}".format(idx)
                uid = job_id * 1000 * 1000 + i * 10 + idx
                uid_list.append(uid)
                tf_record_list.append(tf_record)

        data = zip(tf_record_list, uid_list)
        self.write_tf_record(job_id, data)


    def inspect_seg(self, job_id):
        output_A = self.load_output_A(job_id)
        seg_candi_list = self.load("seg_candi_list",job_id)
        q_path = self.get_path("query", "g_query_{}.json".format(job_id))
        queries = json.load(open(q_path, "r"))["queries"]
        ltr_result = self.load_ltr(job_id)

        for i in range(self.inst_per_job):
            problem, qid = output_A[i]
            print(qid)
            scl = seg_candi_list[i]
            query = queries[i]["text"][len("#combine("):-1]
            print(query)
            q_terms = query.split()
            q_tf = stemmed_counter(q_terms, self.stemmer)
            print(list(q_tf.keys()))
            for doc_id, loc in scl[:30]:
                doc_tokens = self.pr.token_dump.get(doc_id)
                l = self.pr.get_seg_len_dict(doc_id)[loc]
                passage = doc_tokens[loc:loc+l]
                d_tf = stemmed_counter(passage, self.stemmer)


                arr = []

                for qt in q_tf:
                    arr.append(d_tf[qt])
                print(arr)



    def process_A(self):
        segment = self.sample_segment()
        problem = self.segment2problem(segment)
        query = self.problem2query(problem)
        return problem, query

    def process_B(self, problem, doc_candi):
        self.code_tick.tick_begin("get_seg_candidate")
        seg_candi = self.get_seg_candidate(doc_candi, problem)
        self.code_tick.tick_end("get_seg_candidate")
        target_tokens, sent_list, prev_tokens, next_tokens, mask_indice, doc_id = problem
        self.code_tick.tick_begin("get_feature_list")
        feature = self.pr.get_feature_list(doc_id, sent_list, target_tokens, mask_indice, seg_candi)
        self.code_tick.tick_end("get_feature_list")
        return seg_candi, feature

    def process_C(self, problem, seg_candi, scores):
        target_tokens, sent_list, prev_tokens, next_tokens, mask_indice, doc_id = problem

        def select_top(items, scores, n_top):
            return left(list(zip(items, scores)).sort(key=lambda x:x[1], reverse=True)[:n_top])

        e = {
            "target_tokens":target_tokens,
            "sent_list":sent_list,
            "prev_tokens": prev_tokens,
            "next_tokens":next_tokens,
            "mask_indice":mask_indice,
            "doc_id":doc_id,
            "passages": select_top(seg_candi, scores, 3)
        }
        insts = self.tf_record_maker.generate_ir_tfrecord(self, e)

        r = []
        for idx, p in enumerate(insts):
            tf_record = self.tf_record_maker.make_instance(problem)
            r.append((idx, tf_record))
        return r

    def write_tf_record(self, job_id, data):
        inst_list, uid_list = filter_instances(data)
        max_pred = 20
        data = zip(inst_list, uid_list)
        output_path = self.get_path("tf_record_pred", "{}".format(job_id))
        write_predict_instance(data, self.tokenizer, self.model_max_seq, max_pred, [output_path])


    # Use robust_idf_mini
    def sample_segment(self):
        r = get_random_sent()
        s_id, doc_id, loc, g_id, sent= r
        doc_rows = get_doc_sent(doc_id)
        max_seq = self.seg_max_seq
        target_tokens, sent_list, prev_tokens, next_tokens = extend(doc_rows, sent, loc, self.tokenizer, max_seq)
        inst = target_tokens, sent_list, prev_tokens, next_tokens, doc_id
        return inst

    def segment2problem(self, segment):
        mask_inst = generate_mask(segment, self.seg_max_seq, self.masked_lm_prob, self.short_seq_prob, self.rng)
        return mask_inst

    def problem2query(self, problem):
        target_tokens, sent_list, prev_tokens, next_tokens, mask_indice, doc_id = problem
        basic_tokens, word_mask_indice = translate_mask2token_level(sent_list, target_tokens, mask_indice, self.tokenizer)
        # stemming

        visible_tokens = remove(basic_tokens, word_mask_indice)
        stemmed = list([self.stemmer.stem(t) for t in visible_tokens])
        query = self.filter_stopword(stemmed)
        high_qt_stmed = self.pr.high_idf_q_terms(Counter(query))

        final_query = []
        for t in visible_tokens:
            if self.stemmer.stem(t) in high_qt_stmed:
                final_query.append(t)
        return final_query

    def filter_res(self, query_res, doc_id):
        t_doc_id, rank, score = query_res[0]

        assert rank == 1
        if t_doc_id == doc_id:
            valid_res = query_res[1:]
        else:
            valid_res = query_res
            for i in range(len(query_res)):
                e_doc_id, e_rank, _= query_res[i]
                if e_doc_id == t_doc_id:
                    valid_res = query_res[:i] + query_res[i+1:]
                    break
        return valid_res

    def get_seg_candidate(self, doc_candi, problem):
        target_tokens, sent_list, prev_tokens, next_tokens, mask_indice, doc_id = problem
        valid_res = self.filter_res(doc_candi, doc_id)
        return self.pr.rank(doc_id, valid_res, target_tokens, sent_list, mask_indice, top_k=100)


    def filter_stopword(self, l):
        return list([t for t in l if t not in self.stopword])

    def get_path(self, sub_dir_name, file_name):
        out_path = os.path.join(self.iteration_dir, sub_dir_name, file_name)
        dir_path = os.path.join(self.iteration_dir, sub_dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        return out_path

    def save_query(self, job_idx, queries):
        j_queries = []
        for qid, query in queries:
            query = clean_query(query)
            j_queries.append({"number": str(qid), "text": "#combine({})".format(" ".join(query))})


        data = {"queries": j_queries}
        out_path = self.get_path("query", "g_query_{}.json".format(job_idx))
        fout = open(out_path, "w")
        fout.write(json.dumps(data, indent=True))
        fout.close()

    def save_pickle_at(self, obj, sub_dir_name, file_name):
        out_path = self.get_path(sub_dir_name, file_name)
        fout = open(out_path, "wb")
        pickle.dump(obj, fout)

    def load_pickle_at(self, sub_dir_name, file_name):
        out_path = self.get_path(sub_dir_name, file_name)
        fout = open(out_path, "rb")
        return pickle.load(fout)

    def save_output_A(self, job_idx, output):
        self.save_pickle_at(output, "output_A", "{}.pickle".format(job_idx))

    def load_output_A(self, job_idx):
        return self.load_pickle_at("output_A", "{}.pickle".format(job_idx))

    def save(self, dir_name, job_idx, output):
        self.save_pickle_at(output, dir_name, "{}.pickle".format(job_idx))

    def load(self, dir_name, job_idx):
        return self.load_pickle_at(dir_name, "{}.pickle".format(job_idx))

    def load_candidate_docs(self, job_id):
        out_path = self.get_path("q_res", "{}.txt".format(job_id))
        return load_galago_ranked_list(out_path)

    def save_ltr(self, job_idx, data):
        out_path = self.get_path("ltr", str(job_idx))
        s = "\n".join(data)
        open(out_path, "w").write(s)

    def load_ltr(self, job_idx):
        return NotImplemented

def a_runner():
    iter = 1
    pipeline = Pipeline()
    mark_path = os.path.join(pipeline.iteration_dir, "mark", "{}_A".format(iter))
    mtm = MarkedTaskManager(1000*1000, mark_path, 1000)

    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        pipeline.run_A(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)

def b_runner():
    iter = 1
    pipeline = Pipeline()
    max_job = 1000
    mark_path = os.path.join(pipeline.iteration_dir, "mark", "{}_B".format(iter))
    ready_sig = os.path.join(pipeline.iteration_dir, "q_res", "{}.txt")
    mtm = ReadyMarkTaskManager(max_job, ready_sig, mark_path)

    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        pipeline.run_B(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)


def b_benchmark():
    pipeline = Pipeline()
    pipeline.run_B(0)


def main():
    task = sys.argv[1]
    if task == "A":
        a_runner()
    if task == "B":
        b_runner()



if __name__ == "__main__":
    #main()

    b_benchmark()