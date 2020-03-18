import collections

import cpath
from cache import *
from data_generator import tokenizer_wo_tf as tokenization
from list_lib import lmap
from misc_lib import *
from rpc.disk_dump import DumpAccess
from tlm.retrieve_lm.segment_ranker_0 import PassageRanker, get_visible
from tlm.retrieve_lm.stem import stemmed_counter


class LazyLoader:
    def __init__(self, c_path):
        self.c_path = c_path
        self.d ={}

    def load(self, i ):
        self.d[i] = pickle.load(open(self.c_path.format(i), "rb"))
        
    def get(self, i):
        if i in self.d:
            return self.d[i]
        else:
            self.load(i)
            return self.d[i]

        
class FeatureExtractor(PassageRanker):
    def __init__(self, window_size):
        super().__init__(window_size)

        self.date_dict = load_from_pickle("robust_date")
        #self.token_reader = get_token_reader()
        self.token_dump = DumpAccess("robust_token")
        self.text_dump = DumpAccess("robust")
        c_path = os.path.join(data_path, "stream_pickled", "CandiSet_{}_0")
        self.ll = LazyLoader(c_path)
        vocab_file = os.path.join(cpath.data_path, "bert_voca.txt")
        self.cap_tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=False)

    def get_feature_list(self, job_id, idx):
        entry = self.ll.get(job_id)[idx]
        src_doc_id = entry["doc_id"]
        query_tokens = get_visible(entry["sent_list"],
                                     entry["target_tokens"],
                                     entry["mask_indice"],
                                     self.tokenizer)

        query_tokens_cap = get_visible(entry["sent_list"],
                                     entry["target_tokens"],
                                     entry["mask_indice"],
                                     self.cap_tokenizer)

        day_src = self.get_day(src_doc_id)
        q_tf_all = stemmed_counter(query_tokens, self.stemmer)
        q_tf_top = self.reform_query(query_tokens)
        feature_list = []
        for j in range(3):
            doc_id, loc = entry["passages"][j]

            headline = self.head_tokens[doc_id]
            raw_doc = self.text_dump.get(doc_id)
            doc_tokens = self.token_dump.get(doc_id)
            doc_tokens_cap = self.cap_tokenizer.basic_tokenizer.tokenize(raw_doc)

            assert len(doc_tokens_cap) == len(doc_tokens)
            l = self.get_seg_len_dict(doc_id)[loc]
            seg_tokens_cap = doc_tokens_cap[loc:loc + l]
            seg_tokens = doc_tokens[loc:loc + l]
            day_hint = self.get_day(doc_id)
            
            feature = {
                1: self.BM25(q_tf_all, doc_tokens), # All Source segment as query
                2: self.BM25(q_tf_all, seg_tokens), #
                3: self.BM25(q_tf_all, headline),   #
                4: len(doc_tokens),
                5: self.unique_n_gram(query_tokens, seg_tokens, 1),
                6: self.unique_n_gram(query_tokens, seg_tokens, 2),
                7: self.unique_n_gram(query_tokens, seg_tokens, 3),
                8: self.unique_n_gram(query_tokens, seg_tokens, 4),
                9: self.unique_cap_n_gram(query_tokens_cap, seg_tokens_cap, 1),
                10: self.unique_cap_n_gram(query_tokens_cap, seg_tokens_cap, 2),
                11: self.unique_cap_n_gram(query_tokens_cap, seg_tokens_cap, 3),
                12: self.unique_cap_n_gram(query_tokens_cap, seg_tokens_cap, 4),
                13: self.is_same_day(day_src, day_hint),
                14: self.day_diff(day_src, day_hint),
                15: self.BM25(q_tf_top, doc_tokens),
                16: self.BM25(q_tf_top, seg_tokens),
                17: self.BM25(q_tf_top, headline),
                18: loc,
            }
            feature_list.append(feature)
            
        return feature_list

    def get_day(self, doc_id):
        if doc_id in self.date_dict:
            return self.date_dict[doc_id]
        else:
            return None

    def is_same_day(self, day1, day2):
        if day1 is None or day2 is None:
            return 0
        return int(day1 == day2)

    def day_diff(self, day1, day2):
        max_diff = 365
        if day1 is None or day2 is None:
            return max_diff
        return min(abs(day1 - day2).days, max_diff)

    def unique_n_gram(self, q_tokens, doc_tokens, n_gram):
        q1 = self.stemmer.stem_list(q_tokens)
        q2 = self.stemmer.stem_list(doc_tokens)
        voca1 = self.get_n_gram_set(q1, n_gram)
        voca2 = self.get_n_gram_set(q2, n_gram)

        return voca1.intersection(voca2).__len__()

    def get_n_gram_set(self, tokens, n_gram):
        i = 0
        s = set()
        while i + n_gram < len(tokens):
            sig = " ".join(tokens[i:i+n_gram])
            s.add(sig)
            i += 1
        return s

    def unique_cap_n_gram(self, q_tokens, doc_tokens, n_gram):
        def drop_small(tokens):
            r = []
            for t in tokens:
                if t[0].isupper():
                    r.append(t)
            return r

        return self.unique_n_gram(
            drop_small(q_tokens),
            drop_small(doc_tokens),
            n_gram
        )

RawResult = collections.namedtuple("RawResult",
                                   ["unique_ids", "losses"])

def load_run_1():

    info_d = {}
    for job_id in range(5):
        p = os.path.join(cpath.data_path, "tlm", "pred", "info_d_{}.pickle".format(job_id))
        d = pickle.load(open(p, "rb"))
        info_d.update(d)

    p = os.path.join(cpath.data_path, "tlm", "pred", "tlm1.pickle")
    pred = pickle.load(open(p, "rb"))

    tf_id_set = set()

    tf_id_and_loss = []
    for e in pred:
        tf_id = info_d[e.unique_ids]
        if tf_id not in tf_id_set:
            tf_id_set.add(tf_id)
            loss = e.losses
            tf_id_and_loss.append((tf_id, loss))
    return tf_id_and_loss


def libsvm_str(qid, score, feature_d):
    prefix = "{} qid:{} ".format(score, qid)
    postfix = ""
    for key, value in feature_d.items():
        postfix += "{}:{} ".format(key,value)
    return prefix + postfix


def run():
    tf_id_list, loss_list = zip(*load_run_1())
    fe = FeatureExtractor(window_size=256 - 3)

    loss_d = dict(zip(tf_id_list, loss_list))
    print(len(tf_id_list))
    
    feature_list_list = []
    history = set()

    feature_str_list = []
    ticker = TimeEstimator(len(tf_id_list), "TAsk", 100)
    for tf_id in tf_id_list:
        ticker.tick()
        job_id, idx, candi_idx = tf_id.split("_")
        job_id = int(job_id)
        idx = int(idx)
        candi_idx = int(candi_idx)
        if (job_id, idx) in history:
            continue
        print(job_id, idx)
        history.add((job_id, idx))
        try:
            feature_list = fe.get_feature_list(job_id, idx)
            feature_list_list.append(feature_list)

            base_scores = []
            for i in range(2):
                candi_idx = i
                tf_id = "{}_{}_{}".format(job_id, idx, candi_idx)
                loss = loss_d[tf_id]
                print(tf_id, loss)
                base_scores.append(loss)

            worst_score = max(base_scores)

            scores = []
            for i in range(3):
                candi_idx = i+2
                tf_id = "{}_{}_{}".format(job_id, idx, candi_idx)
                loss = loss_d[tf_id]
                print(tf_id, loss)
                scores.append(loss)

            def fn_adj(s):
                p = worst_score - s
                return int(p * 10)
            adj_score = lmap(fn_adj, scores)

            for i in range(3):
                print(scores[i], adj_score[i])

            qid = job_id * 10 + idx
            fstr = "\n".join([libsvm_str(qid, adj_score[i], feature_list[i]) for i in range(3)])
            feature_str_list.append(fstr)

        except ValueError as e:
            print("ValueError")
        except KeyError as e:
            print(tf_id)

    cut = int(len(feature_str_list) * 0.7)
    train_set = "\n".join(feature_str_list[:cut])
    test_set = "\n".join(feature_str_list[cut:])

    out_dir = os.path.join(cpath.data_path, "tlm", "feature")
    open(os.path.join(out_dir, "train.txt"), "w").write(train_set)
    open(os.path.join(out_dir, "test.txt"), "w").write(test_set)


def main():
    run()


if __name__ == "__main__":
    main()

