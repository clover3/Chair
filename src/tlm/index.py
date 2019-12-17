import os
import pickle
from collections import Counter, defaultdict

import nltk.tokenize
from krovetzstemmer import Stemmer

import data_generator.data_parser.trec as trec
import path
from data_generator import tokenizer_b as tokenization
from misc_lib import TimeEstimator
from models.classic.stopword import load_stopwords
from tlm.retrieve_lm.stem import CacheStemmer
from tlm.retrieve_lm.token_server import get_token_reader


def build_krovetz_index():
    stemmer = Stemmer()
    stopwords = load_stopwords()

    stem_dict = dict()

    def stem(token):
        if token in stem_dict:
            return stem_dict[token]
        else:
            r = stemmer.stem(token)
            stem_dict[token] = r
            return r

    collection = trec.load_robust(trec.robust_path)
    print("writing...")
    inv_index = dict()
    ticker = TimeEstimator(len(collection))

    for doc_id in collection:
        content = collection[doc_id]
        tokens = nltk.tokenize.wordpunct_tokenize(content)
        terms = dict()
        for idx, t in enumerate(tokens):
            if t in stopwords:
                continue

            t_s = stem(t)

            if t_s not in terms:
                terms[t_s] = list()

            terms[t_s].append(idx)


        for t_s in terms:
            if t_s not in inv_index:
                inv_index[t_s] = list()

            posting = (doc_id, terms[t_s])
            inv_index[t_s].append(posting)

        ticker.tick()

    save_path = os.path.join(path.data_path, "adhoc", "robust_inv_index.pickle")
    pickle.dump(inv_index, open(save_path, "wb"))


def save_doc_len():
    collection = trec.load_robust(trec.robust_path)
    print("writing...")
    ticker = TimeEstimator(len(collection))

    doc_len = dict()
    for doc_id in collection:
        content = collection[doc_id]
        tokens = nltk.tokenize.wordpunct_tokenize(content)
        doc_len[doc_id] = len(tokens)
        ticker.tick()

    save_path = os.path.join(path.data_path, "adhoc", "doc_len.pickle")
    pickle.dump(doc_len, open(save_path, "wb"))



def save_qdf():
    ii_path = os.path.join(path.data_path, "adhoc", "robust_inv_index.pickle")
    inv_index = pickle.load(open(ii_path, "rb"))
    qdf_d = Counter()
    for term in inv_index:
        qdf = len(inv_index[term])
        qdf_d[term] = qdf

    save_path = os.path.join(path.data_path, "adhoc", "robust_qdf.pickle")
    pickle.dump(qdf_d, open(save_path, "wb"))

def save_qdf_ex():
    ii_path = os.path.join(path.data_path, "adhoc", "robust_inv_index.pickle")
    inv_index = pickle.load(open(ii_path, "rb"))
    save_path = os.path.join(path.data_path, "adhoc", "robust_meta.pickle")
    meta = pickle.load(open(save_path, "rb"))
    stopwords = load_stopwords()
    stemmer = CacheStemmer()

    simple_posting = {}

    qdf_d = Counter()
    for term in inv_index:
        simple_posting[term] = set()
        for doc_id, _ in inv_index[term]:
            simple_posting[term].add(doc_id)

    for doc in meta:
        date, headline = meta[doc]
        tokens = nltk.tokenize.wordpunct_tokenize(headline)
        terms = set()
        for idx, t in enumerate(tokens):
            if t in stopwords:
                continue

            t_s = stemmer.stem(t)

            terms.add(t_s)

        for t in terms:
            simple_posting[t].add(doc)

    for term in inv_index:
        qdf = len(simple_posting[term])
        qdf_d[term] = qdf

    save_path = os.path.join(path.data_path, "adhoc", "robust_qdf_ex.pickle")
    pickle.dump(qdf_d, open(save_path, "wb"))


def save_title():
    collection = trec.load_robust_meta(trec.robust_path)
    save_path = os.path.join(path.data_path, "adhoc", "robust_meta.pickle")
    pickle.dump(collection, open(save_path, "wb"))


def save_title_tokens():
    meta_path = os.path.join(path.data_path, "adhoc", "robust_meta.pickle")
    meta = pickle.load(open(meta_path, "rb"))
    vocab_file = os.path.join(path.data_path, "bert_voca.txt")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    head_tokens = {}
    ticker = TimeEstimator(len(meta))
    for doc_id in meta:
        date, headline = meta[doc_id]
        h_tokens = tokenizer.basic_tokenizer.tokenize(headline)
        head_tokens[doc_id] = h_tokens
        ticker.tick()

    save_path = os.path.join(path.data_path, "adhoc", "robust_title_tokens.pickle")
    pickle.dump(head_tokens, open(save_path, "wb"))



def get_doc_task(task_id):
    doclen_path = os.path.join(path.data_path, "adhoc", "doc_len.pickle")
    doc_len = pickle.load(open(doclen_path, "rb"))
    doc_id_list = list(doc_len.keys())
    doc_id_list.sort()

    n_split = 40
    task_size = int(len(doc_id_list) / n_split) + 1


    st = task_size * task_id
    ed = task_size * (task_id+1)
    return doc_id_list[st:ed]

def subtoken_split(task_id):
    #robust_tokens = load_robust_token()
    token_reader = get_token_reader()

    doc_id_list = get_doc_task(task_id)
    num_doc = len(doc_id_list)


    vocab_file = os.path.join(path.data_path, "bert_voca.txt")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    window_size = 256 - 3

    skip = int(window_size/2)
    ticker = TimeEstimator(num_doc)

    doc_seg_info = {}
    for key in doc_id_list:
        tokens = token_reader.retrieve(key)
        fn = tokenizer.wordpiece_tokenizer.tokenize
        sub_tokens = list([fn(t) for t in tokens])

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

        interval_list = []

        while loc < len(tokens):
            loc_ed, loc_sub_ed = move(loc, loc_sub, skip)
            e = (loc, loc_ed) , (loc_sub, loc_sub_ed)
            interval_list.append(e)
            loc = loc_ed
            loc_sub = loc_sub_ed

        doc_seg_info[key] = interval_list
        ticker.tick()

    p = os.path.join(path.data_path, "adhoc", "robust_seg_info_{}.pickle".format(task_id))
    pickle.dump(doc_seg_info, open(p, "wb"))


def segment_per_doc_index(task_id):
    token_reader = get_token_reader()
    stemmer = CacheStemmer()
    stopword = load_stopwords()

    p = os.path.join(path.data_path, "adhoc", "robust_seg_info.pickle")
    seg_info = pickle.load(open(p, "rb"))
    def get_doc_posting_list(doc_id):
        doc_posting = defaultdict(list)
        for interval in seg_info[doc_id]:
            (loc, loc_ed), (_,_) = interval
            tokens = token_reader.retrieve(doc_id)
            st_tokens = list([stemmer.stem(t) for t in tokens])
            ct = Counter(st_tokens[loc:loc_ed])
            for term, cnt in ct.items():
                if term in stopword:
                    continue
                doc_posting[term].append((loc, cnt))


        return doc_posting

    doc_id_list = get_doc_task(task_id)
    ticker = TimeEstimator(len(doc_id_list))
    doc_posting_d = {}
    for doc_id in doc_id_list:
        doc_posting_d[doc_id] = get_doc_posting_list(doc_id)
        ticker.tick()

    save_path = os.path.join(path.data_path, "adhoc", "per_doc_posting_{}.pickle".format(task_id))
    pickle.dump(doc_posting_d, open(save_path, "wb"))


def merge_sub_pickle(n_task, format_path, save_path):
    all_d = {}
    for i in range(n_task):
        p = format_path.format(i)
        if not os.path.exists(p):
            print("Warning Path not exists : ", p)
            continue
        pdp = pickle.load(open(p, "rb"))
        for key in pdp:
            all_d[key] = pdp[key]

    pickle.dump(all_d, open(save_path, "wb"))

def merge_per_posting():
    n_task = 40
    format_path = os.path.join(path.data_path, "adhoc", "per_doc_posting_{}.pickle")
    save_path = os.path.join(path.data_path, "adhoc", "per_doc_posting.pickle")
    merge_sub_pickle(n_task, format_path, save_path)

def merge_seg_info():
    n_task = 10
    format_path = os.path.join(path.data_path, "adhoc", "robust_seg_info_{}.pickle")
    save_path = os.path.join(path.data_path, "adhoc", "robust_seg_info.pickle")
    merge_sub_pickle(n_task, format_path, save_path)


if __name__ == "__main__":
    #task_id = int(sys.argv[1])
    #print(task_id)
    #merge_seg_info()
    #segment_per_doc_index(task_id)
    merge_per_posting()
    #save_title_tokens()
    #build_krovetz_index()
