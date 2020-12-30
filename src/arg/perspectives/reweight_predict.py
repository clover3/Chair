from collections import Counter
from typing import Dict, List, Tuple, Iterator

import spacy

# num_doc = 541
# avdl = 11.74
from arg.bm25 import BM25
from arg.perspectives.collection_based_classifier import predict_interface
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import claims_to_dict
# num_doc = 541
# avdl = 11.74
from arg.perspectives.runner.claim_analysis import get_claim_term_weighting
from list_lib import dict_value_map
from misc_lib import NamedNumber


def predict_by_reweighter(bm25_module: BM25,
                    claims,
                    top_k,
                    param) -> List[Tuple[str, List[Dict]]]:

    cid_to_text: Dict[int, str] = claims_to_dict(claims)
    claim_term_weight: Dict[int, Dict[str, float]] = get_claim_term_weighting(claims, param)
    nlp = spacy.load("en_core_web_sm")

    def do_stem(t: str) -> str:
        r = bm25_module.tokenizer.stemmer.stem(t)
        return r

    def stem_tokenize(text: str) -> Iterator[str]:
        for t in nlp(text):
            try:
                yield do_stem(t.text)
            except UnicodeDecodeError:
                pass

    def apply_stem(term_weight: Dict[str, float]) -> Dict[str, float]:
        return {do_stem(k): v for k, v in term_weight.items()}

    claim_term_weight : Dict[int, Dict[str, float]] = dict_value_map(apply_stem, claim_term_weight)

    def scorer(lucene_score, query_id) -> NamedNumber:
        claim_id, p_id = query_id.split("_")
        c_text = cid_to_text[int(claim_id)]
        p_text = perspective_getter(int(p_id))
        qtf = Counter(stem_tokenize(c_text))
        weight = claim_term_weight[int(claim_id)]

        new_qtf = Counter()
        for k, v in qtf.items():
            try:
                w = weight[k]
                new_qtf[k] = w * v
            except Exception as e:
                print("Exception")
                print(e)
                print(k)

        tf = Counter(stem_tokenize(p_text))
        score = bm25_module.score_inner(new_qtf, tf)
        return score

    r = predict_interface(claims, top_k, scorer)
    return r

