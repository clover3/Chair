from cache import *
import path
from data_generator import tokenizer_wo_tf as tokenization
from misc_lib import *
from tlm.stem import CacheStemmer, stemmed_counter
from tlm.segment_ranker_1 import PassageRanker, get_visible
from rpc.disk_dump import DumpAccess


def drop_small(tokens):
    r = []
    for t in tokens:
        if t[0].isupper():
            r.append(t)
    return r


class FeatureExtractor(PassageRanker):
    def __init__(self, window_size):
        super().__init__(window_size)

        self.date_dict = load_from_pickle("robust_date")
        # self.token_reader = get_token_reader()
        self.token_dump = DumpAccess("robust_token")
        self.token_dump_cap = DumpAccess("robust_token_cap")
        self.text_dump = DumpAccess("robust")
        c_path = os.path.join(data_path, "stream_pickled", "CandiSet_{}_0")
        vocab_file = os.path.join(path.data_path, "bert_voca.txt")
        self.cap_tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=False)

    def get_feature_list(self, src_doc_id, sent_list, target_tokens, mask_indice, passages):

        before = time.time()
        def tprint(log):
            nonlocal before
            elp = time.time() - before
            print("{} : {}".format(elp * 1000, log))
            before = time.time()

        #tprint("get_feature_list 1")
        query_tokens = get_visible(sent_list,
                                   target_tokens,
                                   mask_indice,
                                   self.tokenizer)
        #tprint("get_feature_list 2")

        query_tokens_cap = get_visible(sent_list,
                                       target_tokens,
                                       mask_indice,
                                       self.cap_tokenizer)
        #tprint("get_feature_list 3")
        day_src = self.get_day(src_doc_id)
        q_tf_all = stemmed_counter(query_tokens, self.stemmer)
        q_tf_top = self.reform_query(query_tokens)
        #tprint("get_feature_list 4")
        feature_list = []


        for passage in passages:
            doc_id, loc = passage
            #tprint("get_feature_list 5")
            headline = self.head_tokens[doc_id]
            raw_doc = self.text_dump.get(doc_id)
            doc_tokens = self.token_dump.get(doc_id)
            #tprint("get_feature_list 6")
            #doc_tokens_cap = self.cap_tokenizer.basic_tokenizer.tokenize(raw_doc)
            doc_tokens_cap = self.token_dump_cap.get(doc_id)

            assert len(doc_tokens_cap) == len(doc_tokens)
            l = self.get_seg_len_dict(doc_id)[loc]
            seg_tokens_cap = doc_tokens_cap[loc:loc + l]
            seg_tokens = doc_tokens[loc:loc + l]
            day_hint = self.get_day(doc_id)
            #tprint("get_feature_list 7")
            feature = {
                1: self.BM25(q_tf_all, doc_tokens), # All Source segment as query
                2: self.BM25(q_tf_all, seg_tokens), #
                3: self.BM25(q_tf_all, headline),   #
                4: len(doc_tokens),
                5: self.unique_n_gram(query_tokens, seg_tokens, 1),
                6: self.unique_n_gram(query_tokens, seg_tokens, 2),
                7: self.unique_n_gram(query_tokens, seg_tokens, 3),
                8: self.unique_n_gram(query_tokens, seg_tokens, 4),
                9:  self.unique_cap_n_gram(query_tokens_cap, seg_tokens_cap, 1),
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
            #tprint("get_feature_list 8")
            feature_list.append(feature)

        return feature_list


    def get_feature(self, q_tf_all, query_tokens, day_src, q_tf_top, doc_id, loc):
        hint_tf_d = NotImplemented
        hint_dl = NotImplemented
        hint_n_gram = NotImplemented
        hint_cap_n_gram = NotImplemented

        def get_info(doc_id, loc):
            doc_tokens = self.token_dump.get(doc_id)
            doc_tokens_cap = self.token_dump_cap.get(doc_id)
            l = self.get_seg_len_dict(doc_id)[loc]
            seg_tokens_cap = doc_tokens_cap[loc:loc + l]
            seg_tokens = doc_tokens[loc:loc + l]
            day_hint = self.get_day(doc_id)

            s1 = self.stemmer.stem_list(seg_tokens)
            s_cap = drop_small(self.stemmer.stem_list(seg_tokens_cap))

            hint_n_gram = {}
            for n in range(1,5):
                hint_n_gram[n] = self.get_n_gram_set(s1, n)
                hint_cap_n_gram[n] = self.get_n_gram_set(s_cap, n)

            return hint_n_gram, hint_cap_n_gram

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
            sig = " ".join(tokens[i: i +n_gram])
            s.add(sig)
            i += 1
        return s

    def unique_cap_n_gram(self, q_tokens, doc_tokens, n_gram):

        return self.unique_n_gram(
            drop_small(q_tokens),
            drop_small(doc_tokens),
            n_gram
        )
