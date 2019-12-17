from collections import Counter

import path
from adhoc.bm25 import BM25_3, BM25_3_q_weight
from adhoc.galago import load_galago_judgement
from cache import *
from data_generator import tokenizer_wo_tf as tokenization
from misc_lib import *
from models.classic.stopword import load_stopwords
from sydney_manager import MarkedTaskManager
from tlm.retrieve_lm import per_doc_posting_server
from tlm.retrieve_lm.retreive_candidates import get_visible
from tlm.retrieve_lm.stem import CacheStemmer, stemmed_counter


class DocRelLoader:
    def __init__(self, dir_path):
        self.cur_chunk_id = -1
        self.cur_chunk = None
        self.fail_record = Counter()
        self.dir_path = dir_path

    def load_chunk(self, chunk_id):
        name = str(chunk_id) + ".txt"
        chunk_path = os.path.join(self.dir_path, name)

        if not os.path.exists(chunk_path):
            print("Not exists : ")
            print(chunk_path)

        self.cur_chunk = load_galago_judgement(chunk_path)
        self.cur_chunk_id = chunk_id


    def get_rel_doc_list(self, g_idx, q_id):
        chunk_id = int(g_idx / 10000)
        if self.cur_chunk_id != chunk_id:
            self.load_chunk(chunk_id)

        qid_str = str(q_id)

        if qid_str in self.cur_chunk:
            r = self.cur_chunk[qid_str]
            r.sort(key=lambda x:x[1])
            return r

        else:
            self.fail_record[chunk_id] += 1
            if self.fail_record[chunk_id] > 25:
                print("Too many fail on {}".format(chunk_id))
            return None


class PassageRanker:
    def __init__(self, window_size):
        self.stemmer = CacheStemmer()
        self.window_size = window_size
        self.doc_posting = None
        self.stopword = load_stopwords()

        vocab_file = os.path.join(path.data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)


        def load_pickle(name):
            p = os.path.join(path.data_path, "adhoc", name + ".pickle")
            return pickle.load(open(p, "rb"))

        self.doc_len_dict = load_pickle("doc_len")
        self.qdf = load_pickle("robust_qdf_ex")
        self.meta = load_pickle("robust_meta")
        self.head_tokens = load_pickle("robust_title_tokens")
        self.seg_info = load_pickle("robust_seg_info")
        self.not_found = set()

        self.total_doc_n =  len(self.doc_len_dict)
        self.avdl = sum(self.doc_len_dict.values()) / len(self.doc_len_dict)
        tprint("Init PassageRanker")


    def high_idf_q_terms(self, q_tf, n_limit=10):
        total_doc = self.total_doc_n

        high_qt = Counter()
        for term, qf in q_tf.items():
            qdf = self.qdf[term]
            w = BM25_3_q_weight(qf, qdf, total_doc)
            high_qt[term] = w

        return set(left(high_qt.most_common(n_limit)))

    def get_seg_len_dict(self, doc_id):
        interval_list = self.seg_info[doc_id]
        l_d = {}
        for e in interval_list:
            (loc, loc_ed), (loc_sub, loc_sub_ed)  = e
            l_d[loc] = loc_ed - loc
        return l_d


    def filter_stopword(self, l):
        return list([t for t in l if t not in self.stopword])

    def BM25(self, q_tf, tokens):
        dl = len(tokens)
        tf_d = stemmed_counter(tokens, self.stemmer)
        score = 0
        for term, qf in q_tf.items():
            tf = tf_d[term]
            qdf = self.qdf[term]
            total_doc = self.total_doc_n
            score += BM25_3(tf, qf, qdf, total_doc, dl, self.avdl)
        return score

    def BM25_seg(self, term, q_tf, tf_d, dl):
        qdf = self.qdf[term]
        total_doc = self.total_doc_n
        return BM25_3(tf_d, q_tf, qdf, total_doc, dl, self.avdl)


    def high_idf_q_terms(self, q_tf, n_limit=10):
        total_doc = self.total_doc_n

        high_qt = Counter()
        for term, qf in q_tf.items():
            qdf = self.qdf[term]
            w = BM25_3_q_weight(qf, qdf, total_doc)
            high_qt[term] = w

        return set(left(high_qt.most_common(n_limit)))


    def reform_query(self, tokens):
        stemmed = list([self.stemmer.stem(t) for t in tokens])
        query = self.filter_stopword(stemmed)
        q_tf = Counter(query)
        high_qt = self.high_idf_q_terms(q_tf)
        new_qf = Counter()
        for t in high_qt:
            new_qf[t] = q_tf[t]
        return new_qf

    def get_title_tokens(self, doc_id):
        return self.head_tokens[doc_id]

    def weight_doc(self, title, content):
        new_doc = []
        title_repeat = 3
        content_repeat = 1
        for t in title:
            new_doc += [t] * title_repeat
        for t in content:
            new_doc += [t] * content_repeat
        return new_doc

    def pick_diverse(self, candidate, top_k):
        doc_set = set()
        r = []
        for j in range(len(candidate)):
            score, window_subtokens, window_tokens, doc_id, loc = candidate[j]
            if doc_id not in doc_set:
                r.append(candidate[j])
                doc_set.add(doc_id)
                if len(r) == top_k:
                    break
        return r




    def rank(self, doc_id, query_res, target_tokens, sent_list, mask_indice, top_k):
        if self.doc_posting is None:
            self.doc_posting = per_doc_posting_server.load_dict()
        title = self.get_title_tokens(doc_id)
        visible_tokens = get_visible(sent_list, target_tokens, mask_indice, self.tokenizer)

        source_doc_rep = self.weight_doc(title, visible_tokens)
        q_tf = self.reform_query(source_doc_rep)

        # Rank Each Window by BM25F
        stop_cut = 100
        segments = Counter()

        n_not_found = 0

        for doc_id, rank, doc_score in query_res[:stop_cut]:
            doc_title = self.get_title_tokens(doc_id)
            title_tf = stemmed_counter(doc_title, self.stemmer)
            per_doc_index = self.doc_posting[doc_id] if doc_id in self.doc_posting else None
            seg_len_d = self.get_seg_len_dict(doc_id)

            if per_doc_index is None:
                self.not_found.add(doc_id)
                n_not_found += 1
                continue

            assert doc_id not in self.not_found

            # Returns location after moving subtoken 'skip' times

            for term, qf in q_tf.items():
                if term not in per_doc_index:
                    continue
                for seg_loc, tf in per_doc_index[term]:
                    rep_tf = tf + title_tf[term]
                    dl = seg_len_d[seg_loc] + len(doc_title)
                    segments[(doc_id,seg_loc)] += self.BM25_seg(term, qf, rep_tf, dl)

        r = []
        for e , score in segments.most_common(top_k):
            if score == 0:
                print("Doc {} has 0 score".format(doc_id))
            r.append(e)
        if n_not_found > 0.9 * len(query_res):
            print("WARNING : {} of {} not found".format(n_not_found, len(query_res)))
        return r


def do_seg_rank(pr, dr, job_id):
    task_size = 1000
    start_idx = job_id
    g_idx = job_id * task_size
    sp_prob_q = StreamPickleReader("robust_problem_q_", start_idx)
    sp = StreamPickler("CandiSet_{}_".format(job_id), 1000)
    ticker = TimeEstimator(task_size)
    top_k = 3

    # Iterate for 1000 instances
    while sp_prob_q.limited_has_next(task_size):
        inst, qid = sp_prob_q.get_item()
        target_tokens, sent_list, prev_tokens, next_tokens, mask_indice, doc_id = inst
        query_res = dr.get_rel_doc_list(g_idx, qid)

        if query_res is not None:
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

            tprint(qid)
            passages = pr.rank(doc_id, valid_res, target_tokens, sent_list, mask_indice, top_k)

            if len(passages) < top_k :
                print("Skip qid : ", qid)
                continue

            entry = {
                "target_tokens" : target_tokens,
                "sent_list" : sent_list,
                "prev_tokens" : prev_tokens,
                "next_tokens" : next_tokens,
                "mask_indice" : mask_indice,
                "doc_id" : doc_id,
                "passages" : passages
            }
            sp.add(entry)

        g_idx += 1
        ticker.tick()
    sp.flush()

def main():
    print("Segment_ranker_0")
    mark_path = os.path.join(path.data_path, "adhoc", "seg_rank0_mark")
    mtm = MarkedTaskManager(1000*1000, mark_path, 1000)
    pr = PassageRanker(256 - 3)
    dir_path = os.path.join(path.data_path, "tlm_res")
    dr = DocRelLoader(dir_path)

    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        do_seg_rank(pr, dr, job_id)
        job_id = mtm.pool_job()


if __name__ == "__main__":
    main()
