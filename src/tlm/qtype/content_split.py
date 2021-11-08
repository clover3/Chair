from collections import Counter
from math import log

import spacy

from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from dataset_specific.msmarco.common import load_queries
from list_lib import lmap
from tlm.qtype.is_functionword import FunctionWordClassifier


def calc_idf(N, df):
    return log((N-df+0.5)/(df + 0.5))

def dev_demo():
    cls = FunctionWordClassifier()
    queries = load_queries("train")
    nlp = spacy.load("en_core_web_sm")
    tf, df = load_clueweb12_B13_termstat()
    df_clue = max(df.values()) * 2
    # nlp.add_pipe("merge_noun_chunks")
    df_msmarco = len(queries)
    unseen = set()

    def token_scoring(raw_token):
        token = str(raw_token).lower()
        if token not in cls.qdf and token not in unseen:
            print("{} is not found".format(token))
            unseen.add(token)

        idf1 = calc_idf(df_msmarco, cls.qdf[token])
        idf2 = calc_idf(df_clue, df[token])
        # print(token, idf1, idf2)
        return idf1 + idf2

    counter = Counter()
    threshold = 10
    print(len(queries))
    for qid, q_str in queries[:10000]:
        q_str = q_str.strip()
        tokens = nlp(q_str)
        score_list = lmap(token_scoring, tokens)

        st = None
        ed = None
        for idx, score in enumerate(score_list):
            if score > threshold:
                if st is None:
                    st = idx
                ed = idx+1

        if st is None:
            parsed_output = ">> Content word not found"
            # max_score_idx = find_max_idx(score_list, lambda x: -x)
        else:
            content_tokens = []
            functional_tokens = []
            out_s_list = []
            for idx, raw_t in enumerate(tokens):
                t = str(raw_t)
                if idx == st:
                    out_s_list.append("[")
                out_s_list.append(t)

                if idx+1 == ed:
                    out_s_list.append("]")

                if st <= idx < ed:
                    content_tokens.append(t)
                else:
                    functional_tokens.append(t)

            parsed_output = " ".join(out_s_list)

            func_str = " ".join(functional_tokens)

            counter[func_str]  += 1
        # print(parsed_output)
        # print(score_list)

    for s, cnt in counter.most_common():
        print(s, cnt)


if __name__ == "__main__":
    dev_demo()
