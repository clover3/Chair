from collections import Counter
from typing import Dict, List, Tuple

from arg.perspectives.collection_based_classifier import predict_interface, NamedNumber
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import claims_to_dict
from arg.perspectives.pc_tokenizer import PCTokenizer
# num_doc = 541
# avdl = 11.74
from cache import load_from_pickle, save_to_pickle
from list_lib import dict_value_map, lmap
# num_doc = 541
# avdl = 11.74
from models.classic.bm25 import BM25


def get_bm25_module():
    df = load_from_pickle("pc_df")
    return BM25(df, avdl=11.7, num_doc=541 + 400, k1=0.00001, k2=100, b=0.5)


def get_bm25_module_no_idf():
    df = Counter()
    return BM25(df, avdl=11.7, num_doc=541 + 400, k1=0.00001, k2=100, b=0.5)


def predict_by_bm25(bm25_module,
                    claims,
                    top_k) -> List[Tuple[str, List[Dict]]]:

    cid_to_text: Dict[int, str] = claims_to_dict(claims)

    def scorer(lucene_score, query_id) -> NamedNumber:
        claim_id, p_id = query_id.split("_")
        c_text = cid_to_text[int(claim_id)]
        p_text = perspective_getter(int(p_id))
        score = bm25_module.score(c_text, p_text)
        return score

    r = predict_interface(claims, top_k, scorer)
    return r


def normalize_scores(score_list: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    if not score_list:
        return score_list
    nonzero_min = min([v for k, v in score_list if v > 0])
    score_list = [(k, v) if v > 0 else (k, nonzero_min * 0.2) for k, v in score_list ]
    return lmap(lambda x: (x[0], x[1] * 50000), score_list)


def parse_float(score_list: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
    return list([(k, float(v)) for k, v in score_list])


def predict_by_bm25_rm(bm25_module: BM25,
                       rm_info: Dict[str, List[Tuple[str, str]]],
                       claims,
                       top_k) -> List[Tuple[str, List[Dict]]]:

    cid_to_text: Dict[int, str] = claims_to_dict(claims)
    tokenizer = PCTokenizer()

    def stem_merge(score_list: List[Tuple[str, float]]) -> Counter:
        c = Counter()
        for k, v in score_list:
            try:
                new_k = tokenizer.stemmer.stem(k)
                c[new_k] += v
            except UnicodeDecodeError:
                pass
        return c

    rm_info: Dict[str, List[Tuple[str, float]]] = dict_value_map(parse_float, rm_info)
    rm_info: Dict[str, List[Tuple[str, float]]] = dict_value_map(normalize_scores, rm_info)
    rm_info_c: Dict[str, Counter] = dict_value_map(stem_merge, rm_info)
    print(len(rm_info_c.keys()))
    print(len(claims))
    not_found = set()
    def scorer(lucene_score, query_id) -> NamedNumber:
        claim_id, p_id = query_id.split("_")
        c_text = cid_to_text[int(claim_id)]
        p_text = perspective_getter(int(p_id))
        score: NamedNumber = bm25_module.score(c_text, p_text)

        nclaim_id = int(claim_id)
        if nclaim_id in rm_info:
            ex_qtf = rm_info_c[nclaim_id]
            p_tokens = tokenizer.tokenize_stem(p_text)
            ex_score = bm25_module.score_inner(ex_qtf, Counter(p_tokens))
            new_info = score.name + "({})".format(ex_score.name)
            score = NamedNumber(score+ex_score, new_info)
        else:
            not_found.add(claim_id)
        return score
    r = predict_interface(claims, top_k, scorer)
    print(not_found)
    return r


def kernel(norm_max, val):
    if val > 0.52 * norm_max:
        return 1
    else:
        return 0


def get_classifier(param):
    bm25 = get_bm25_module()
    verbose = param['verbose']

    def classify(c_text, p_text):
        score_ideal = bm25.score(c_text, c_text)
        cur_score = bm25.score(c_text, p_text)
        return kernel(score_ideal, cur_score)

    return classify



def tune_kernel_save(r):
    bm25 = get_bm25_module()
    data: List[Tuple[float, float, int]] = []
    for c_text, p_text, y in r:
        score_ideal = bm25.score(c_text, c_text)
        cur_score = bm25.score(c_text, p_text)
        data.append((float(score_ideal), float(cur_score), y))

    save_to_pickle(data, "bm25_tune_kernel")



