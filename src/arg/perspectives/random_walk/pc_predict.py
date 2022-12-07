from collections import Counter
from typing import Dict, List, Tuple, Iterator

import spacy

from adhoc.bm25_class import BM25
from arg.perspectives.collection_based_classifier import predict_interface
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import claims_to_dict
from arg.perspectives.runner.claim_analysis import get_claim_term_weighting
from list_lib import dict_value_map
from misc_lib import NamedNumber


def normalize_counter(c: Counter) -> Counter:
    max_val = max(c.values())
    factor = 1 / max_val if max_val > 0 else 1
    out_c = Counter()
    for key, value in c.items():
        out_c[key] = value * factor

    return out_c


def pc_predict_from_vector_query(bm25_module: BM25,
                                 q_tf_replace: Dict[int, Counter],
                                claims,
                                top_k) -> List[Tuple[str, List[Dict]]]:

    cid_to_text: Dict[int, str] = claims_to_dict(claims)
    found_claim = set()
    q_tf_replace_norm = dict_value_map(normalize_counter, q_tf_replace)

    c_qtf_d = {}
    for cid, c_text in cid_to_text.items():
        c_tokens = bm25_module.tokenizer.tokenize_stem(c_text)
        c_qtf_d[cid] = Counter(c_tokens)

    def scorer(lucene_score, query_id) -> NamedNumber:
        nonlocal found_claim
        claim_id, p_id = query_id.split("_")
        i_claim_id = int(claim_id)
        if i_claim_id in q_tf_replace_norm:
            claim_qtf = Counter(dict_value_map(lambda x: x*1, c_qtf_d[i_claim_id]))
            ex_qtf = q_tf_replace_norm[i_claim_id]
            ex_qtf = Counter(dict(ex_qtf.most_common(50)))
            qtf = ex_qtf + claim_qtf
            found_claim.add(i_claim_id)
        else:
            qtf = c_qtf_d[i_claim_id]
        p_text = perspective_getter(int(p_id))
        p_tokens = bm25_module.tokenizer.tokenize_stem(p_text)
        score = bm25_module.score_inner(qtf, Counter(p_tokens))
        return score

    r = predict_interface(claims, top_k, scorer)
    print("{} of {} found".format(len(found_claim), len(claims)))
    return r


def pc_predict_vector_query_and_reweight(bm25_module: BM25,
                                 q_tf_replace: Dict[int, Counter],
                                claims,
                                top_k,
                                param
                                         ) -> List[Tuple[str, List[Dict]]]:

    cid_to_text: Dict[int, str] = claims_to_dict(claims)
    found_claim = set()
    q_tf_replace_norm = dict_value_map(normalize_counter, q_tf_replace)

    def do_stem(t: str) -> str:
        r = bm25_module.tokenizer.stemmer.stem(t)
        return r

    def apply_stem(term_weight: Dict[str, float]) -> Dict[str, float]:
        return {do_stem(k): v for k, v in term_weight.items()}

    claim_term_weight: Dict[int, Dict[str, float]] = get_claim_term_weighting(claims, param)
    claim_term_weight: Dict[int, Dict[str, float]] = dict_value_map(apply_stem, claim_term_weight)

    nlp = spacy.load("en_core_web_sm")
    def stem_tokenize(text: str) -> Iterator[str]:
        for t in nlp(text):
            try:
                yield do_stem(t.text)
            except UnicodeDecodeError:
                pass

    def get_qtf(claim_id):
        weight = claim_term_weight[claim_id]
        new_qtf = Counter()
        c_text = cid_to_text[int(claim_id)]
        qtf = Counter(stem_tokenize(c_text))
        print(weight)
        for k, v in qtf.items():
            try:
                if k in weight:
                    w = weight[k]
                    new_qtf[k] = w * v
                else:
                    new_qtf[k] = v
            except Exception as e:
                print("Exception")
                print(e)
                print(k)
        return new_qtf

    c_qtf_d = {k: get_qtf(k) for k in cid_to_text.keys()}

    # for cid, c_text in cid_to_text.items():
    #     c_tokens = bm25_module.tokenizer.tokenize_stem(c_text)
    #     c_qtf_d[cid] = Counter(c_tokens)

    def scorer(lucene_score, query_id) -> NamedNumber:
        nonlocal found_claim
        claim_id, p_id = query_id.split("_")
        i_claim_id = int(claim_id)
        if i_claim_id in q_tf_replace_norm:
            ex_qtf = q_tf_replace_norm[i_claim_id]
            ex_qtf = Counter(dict(ex_qtf.most_common(50)))
            qtf = ex_qtf + c_qtf_d[i_claim_id]
            found_claim.add(i_claim_id)
        else:
            qtf = c_qtf_d[i_claim_id]
        p_text = perspective_getter(int(p_id))
        p_tokens = bm25_module.tokenizer.tokenize_stem(p_text)
        score = bm25_module.score_inner(qtf, Counter(p_tokens))
        return score

    r = predict_interface(claims, top_k, scorer)
    print("{} of {} found".format(len(found_claim), len(claims)))
    return r



