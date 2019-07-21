from adhoc.galago import load_galago_judgement
from cache import *
import path
from collections import Counter
from data_generator import tokenizer_b as tokenization
from tlm.retreive_candidates import get_visible
from rpc.text_reader import TextReaderClient, dummy_doc
from tlm.token_server import get_token_reader
from misc_lib import *
from tlm.index import CacheStemmer, stemmed_counter
from models.classic.stopword import load_stopwords
from adhoc.bm25 import BM25_3, BM25_3_q_weight

class DocRelLoader:
    def __init__(self):
        self.cur_chunk_id = -1
        self.cur_chunk = None
        self.fail_record = Counter()

    def load_chunk(self, chunk_id):
        dir_path = os.path.join(path.data_path, "tlm_res")
        name = str(chunk_id) + ".txt"
        chunk_path = os.path.join(dir_path, name)

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
        vocab_file = os.path.join(path.data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.text_reader = TextReaderClient()
        self.token_reader = get_token_reader()
        self.stopword = load_stopwords()

        dl_path = os.path.join(path.data_path, "adhoc", "doc_len.pickle")
        self.doc_len_dict = pickle.load(open(dl_path, "rb"))
        self.total_doc_n =  len(self.doc_len_dict)
        self.avdl = sum(self.doc_len_dict.values()) / len(self.doc_len_dict)

        qdf_path = os.path.join(path.data_path, "adhoc", "robust_qdf_ex.pickle")
        self.qdf = pickle.load(open(qdf_path, "rb"))

        meta_path = os.path.join(path.data_path, "adhoc", "robust_meta.pickle")
        self.meta = pickle.load(open(meta_path, "rb"))
        tprint("Init PassageRanker")

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
        date, headline = self.meta[doc_id]
        return self.tokenizer.basic_tokenizer.tokenize(headline)

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
        title = self.get_title_tokens(doc_id)
        visible_tokens = get_visible(sent_list, target_tokens, mask_indice, self.tokenizer)

        source_doc_rep = self.weight_doc(title, visible_tokens)
        q_tf = self.reform_query(source_doc_rep)

        candidate = []
        top_score = query_res[0][2]
        # Rank Each Window by BM25F
        stop_cut = 100
        for doc_id, rank, doc_score in query_res[:stop_cut]:
            #doc_text = self.text_reader.retrieve(doc_id)
            #content_tokens = self.tokenizer.basic_tokenizer.tokenize(doc_text)
            content_tokens = self.token_reader.retrieve(doc_id)
            doc_title = self.get_title_tokens(doc_id)

            fn = self.tokenizer.wordpiece_tokenizer.tokenize
            sub_tokens = list([fn(t) for t in content_tokens])
            assert len(sub_tokens) == len(content_tokens)

            # Returns location after moving subtoken 'skip' times
            def move(loc, loc_sub, skip):
                loc_idx = loc
                num_passed_sw = 0
                loc_sub_idx = loc_sub
                for i in range(skip):
                    num_passed_sw += 1
                    loc_sub_idx += 1
                    if num_passed_sw == len(sub_tokens[loc_idx]):
                        loc_idx += 1
                        num_passed_sw = 0

                    if loc_idx >= len(sub_tokens):
                        break
                # only move in token level
                if num_passed_sw > 0:
                    loc_sub_idx -= num_passed_sw
                return loc_idx, loc_sub_idx

            loc = 0
            loc_sub = 0
            # loc_sub and loc points to same start of token

            skip = int(0.5 * self.window_size)


            while loc < len(content_tokens):
                loc_ed, loc_sub_ed = move(loc, loc_sub, skip)
                #window_tokens = content_tokens[loc:loc_ed]
                #window_subtokens = flatten(sub_tokens[loc:loc_ed])

                #seg_rep = self.weight_doc(doc_title, window_tokens)
                #score = self.BM25(q_tf, seg_rep)

                score = 0
                window_tokens = []
                window_subtokens = [[]]
                candidate.append((score, window_subtokens, window_tokens, doc_id, loc))

                loc = loc_ed
                loc_sub = loc_sub_ed

        print(len(candidate))
        candidate.sort(key=lambda x:x[0], reverse=True)
        for j in range(top_k):
            score, window_subtokens, window_tokens, doc_id, loc = candidate[j]

        return self.pick_diverse(candidate, top_k)


def main():
    start_idx = 0
    sp_prob_q = StreamPickleReader("robust_problem_q_", start_idx)
    pr = PassageRanker(256 - 3 )
    g_idx = 0
    dr = DocRelLoader()
    sp = StreamPickler("CandiSet_{}_".format(start_idx), 1000)
    ticker = TimeEstimator(1000*1000)
    while sp_prob_q.has_next():
        inst, qid = sp_prob_q.get_item()
        target_tokens, sent_list, prev_tokens, next_tokens, mask_indice, doc_id = inst
        query_res = dr.get_rel_doc_list(g_idx, qid)

        if query_res is not None:
            t_doc_id, rank, score = query_res[0]
            assert rank == 1
            if t_doc_id != doc_id:
                for i in range(len(query_res)):
                    e_doc_id, e_rank, _= query_res[i]
                    if e_doc_id == t_doc_id:
                        print(query_res[i])

            tprint(qid)
            passages = pr.rank(doc_id, query_res[1:], target_tokens, sent_list, mask_indice, 3)

            passage_d = []
            for p in passages:
                score, window_subtokens, window_tokens, doc_id, loc = p
                passage_d.append({
                    "score":score,
                    "window_subtokens":window_subtokens,
                    "window_tokens":window_tokens,
                    "doc_id":doc_id,
                    "loc":loc
                })

            entry = {
                "target_tokens" : target_tokens,
                "sent_list" : sent_list,
                "prev_tokens" : prev_tokens,
                "next_tokens" : next_tokens,
                "mask_indice" : mask_indice,
                "doc_id" : doc_id,
                "passages" : passage_d
            }
            sp.add(entry)

        g_idx += 1
        ticker.tick()
    sp.flush()

if __name__ == "__main__":
    main()
