from collections import Counter
from typing import Dict, List, Tuple

import math
from krovetzstemmer import Stemmer

from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.collection_based_classifier import predict_interface, NamedNumber
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import claims_to_dict
from arg.perspectives.pc_tokenizer import PCTokenizer
# num_doc = 541
# avdl = 11.74
from list_lib import dict_value_map


# num_doc = 541
# avdl = 11.74


def load_collection_tf():
    tf, df = load_clueweb12_B13_termstat()
    return tf


def term_odd(token, C, BG, C_ctf, BG_ctf, smoothing ):
    def count(LM, token):
        if token in LM:
            return LM[token]
        else:
            return 0

    tf_c = count(C, token)
    tf_bg = count(BG, token)
    if tf_c == 0 and tf_bg == 0:
        return 0
    P_w_C = tf_c / C_ctf
    P_w_BG = tf_bg / BG_ctf
    if P_w_BG == 0 :
        return 0

    assert P_w_BG > 0
    logC = math.log(P_w_C * smoothing + P_w_BG * (1 - smoothing))
    logNC = math.log(P_w_BG)
    assert (math.isnan(logC) == False)
    assert (math.isnan(logNC) == False)
    return logC - logNC


def stem_counter(c: Dict[str, int], stem):
    out_c = Counter()
    for key, value in c.items():
        try:
            out_c[stem(key)] += value
        except:
            pass
    return out_c


def predict_by_lm(claim_tf: Dict[int, Counter],
                  bg_tf: Dict[str, int],
                  bm25_module,
                  claims,
                  top_k) -> List[Tuple[str, List[Dict]]]:
    cid_to_text: Dict[int, str] = claims_to_dict(claims)

    stemmer = Stemmer()
    claim_tf_stemmed = dict_value_map(
        lambda x: stem_counter(x, stemmer.stem),
        claim_tf)

    c_ctf = dict_value_map(lambda d: sum(d.values()), claim_tf)
    bg_tf_stemmed = stem_counter(bg_tf, stemmer.stem)
    bg_ctf = sum(bg_tf.values())

    tokenizer = PCTokenizer()
    not_found = set()
    def scorer(lucene_score, query_id) -> NamedNumber:
        claim_id, p_id = query_id.split("_")
        p_text = perspective_getter(int(p_id))
        nclaim_id = int(claim_id)
        if nclaim_id in claim_tf_stemmed:
            tokens: List[str] = tokenizer.tokenize_stem(p_text)
            def term_odd_fn(t):
                return term_odd(t,
                         claim_tf_stemmed[nclaim_id],
                         bg_tf_stemmed,
                         c_ctf[nclaim_id],
                         bg_ctf,
                         0.7
                         )

            token_score = list([term_odd_fn(t) for t in tokens])
            name = " ".join(["{0} {1:.2f}".format(t, term_odd_fn(t)) for t in tokens])
            score: NamedNumber = NamedNumber(sum(token_score), name)
        else:
            c_text = cid_to_text[nclaim_id]
            score = bm25_module.score(c_text, p_text)
        return score

    r = predict_interface(claims, top_k, scorer)
    print(not_found)
    return r




if __name__ == "__main__":
    load_collection_tf()

