import os
from collections import Counter
from math import log

import spacy

from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from cache import save_to_pickle, load_pickle_from
from dataset_specific.msmarco.common import load_queries
from epath import job_man_dir
from list_lib import lmap
from misc_lib import TEL
from tlm.qtype.is_functionword import FunctionWordClassifier


def calc_idf(N, df):
    return log((N-df+0.5)/(df + 0.5))



def categorize_token(tokens, st, ed):
    content_tokens = []
    functional_tokens = []
    out_s_list = []
    for idx, raw_t in enumerate(tokens):
        t = str(raw_t)
        if idx == st:
            out_s_list.append("[")
        out_s_list.append(t)
        if idx + 1 == ed:
            out_s_list.append("]")
        if st <= idx < ed:
            content_tokens.append(t)
        else:
            functional_tokens.append(t)

    content_span = str(tokens[st:ed])
    return content_span, functional_tokens, out_s_list


class ContentParser:
    def __init__(self, num_query_msmarco):
        self.num_query_msmarco = num_query_msmarco
        self.unseen = set()
        self.cls = FunctionWordClassifier()
        tf, df = load_clueweb12_B13_termstat()
        self.df_clue = df
        self.num_d_clue = max(df.values()) * 2
        self.threshold = 10

    def token_scoring(self, spacy_token):
        token = str(spacy_token).lower()
        if token not in self.cls.qdf and token not in self.unseen:
            # print("{} is not found".format(token))
            self.unseen.add(token)

        idf1 = calc_idf(self.num_query_msmarco, self.cls.qdf[token])
        idf2 = calc_idf(self.num_d_clue, self.df_clue[token])
        return idf1 + idf2

    def get_content_span(self, spacy_tokens):
        score_list = lmap(self.token_scoring, spacy_tokens)
        st = None
        ed = None
        for idx, score in enumerate(score_list):
            if score > self.threshold:
                if st is None:
                    st = idx
                ed = idx + 1
        return st, ed


def dev_demo():
    queries = load_queries("train")
    parser = ContentParser(len(queries))
    nlp = spacy.load("en_core_web_sm")
    print(len(queries))
    for qid, q_str in TEL(queries):
        q_str = q_str.strip()
        spacy_tokens = nlp(q_str)
        continue


def parse_save_content_tokens(qid_query_tokens_list):
    queries = load_queries("train")
    parser = ContentParser(len(queries))
    parse_failed_qid = []
    parsed_queries = []
    func_tokens_counter = Counter()
    for qid, query, spacy_tokens in TEL(qid_query_tokens_list):
        st, ed = parser.get_content_span(spacy_tokens)
        if st is None:
            parsed_output = ">> Content word not found"
            parse_failed_qid.append(qid)
            st = 0
            ed = len(spacy_tokens)
        content_span, functional_tokens, out_s_list = categorize_token(spacy_tokens, st, ed)
        parsed_output = " ".join(out_s_list)
        func_str = " ".join(functional_tokens)
        d = {
            'qid': qid,
            'query': query,
            'content_span': content_span,
            'functional_tokens': functional_tokens,
            'out_s_list': out_s_list
        }
        func_tokens_counter[func_str] += 1
        parsed_queries.append(d)

    save_obj = parsed_queries, func_tokens_counter
    return save_obj


def _run_query_parsing(num_jobs, split):
    qid_query_tokens_list = []
    for i in range(num_jobs):
        pickle_path = os.path.join(job_man_dir, "msmarco_spacy_query_parse_{}".format(split), "{}".format(i))
        qid_query_tokens_list.extend(load_pickle_from(pickle_path))
    save_obj = parse_save_content_tokens(qid_query_tokens_list)
    save_to_pickle(save_obj, "mmd_query_parse_{}".format(split))



def _run_query_parsing_debug(num_jobs, split):
    qid_query_tokens_list = []
    key = '1000625'
    for i in range(num_jobs):
        pickle_path = os.path.join(job_man_dir, "msmarco_spacy_query_parse_{}".format(split), "{}".format(i))
        l = load_pickle_from(pickle_path)
        if key in [r[0] for r in l]:
            print("key found at", i)
        print("{} loaded from {}".format(len(l), i))
        qid_query_tokens_list.extend(l)
    save_obj = parse_save_content_tokens(qid_query_tokens_list)
    save_to_pickle(save_obj, "mmd_query_parse_{}".format(split))


def run_query_parsing_train():
    num_jobs = 8
    split = "train"
    _run_query_parsing_debug(num_jobs, split)
    # _run_query_parsing(num_jobs, split)


def run_query_parsing_dev():
    num_jobs = 2
    split = "dev"
    _run_query_parsing(num_jobs, split)


def run_query_parsing_test():
    num_jobs = 1
    split = "test"
    _run_query_parsing(num_jobs, split)


if __name__ == "__main__":
    run_query_parsing_train()
    # run_query_parsing_dev()
    # run_query_parsing_test()
