import cpath
from adhoc.bm25 import BM25_3, BM25_3_q_weight
from cache import *
from data_generator import tokenizer_wo_tf as tokenization
from misc_lib import *
from models.classic.stopword import load_stopwords
from tlm.retrieve_lm import per_doc_posting_server
from tlm.retrieve_lm.retreive_candidates import get_visible
from tlm.retrieve_lm.stem import CacheStemmer, stemmed_counter


class PassageRanker:
    def __init__(self, window_size):
        self.stemmer = CacheStemmer()
        self.window_size = window_size
        self.doc_posting = None
        self.stopword = load_stopwords()

        vocab_file = os.path.join(cpath.data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)


        def load_pickle(name):
            p = os.path.join(cpath.data_path, "adhoc", name + ".pickle")
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
