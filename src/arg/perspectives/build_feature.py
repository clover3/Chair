import collections
import string
from typing import List, Counter, Dict, Callable, Tuple

import math
import nltk

from arg.claim_building.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.basic_analysis import PerspectiveCandidate
from arg.perspectives.collection_interface import CollectionInterface
from arg.perspectives.ranked_list_interface import DynRankedListInterface
from datastore.interface import load_multiple
from datastore.table_names import CluewebDocTF
from galagos.types import GalagoDocRankEntry
from list_lib import lmap, dict_value_map, lfilter, lmap_w_exception


# Given claim,perspective pair, generate lexical feature from collection info


def average_tf_over_docs(docs_rel_freq: List[Counter], num_doc: int) -> Counter:
    p_w_m = collections.Counter()
    for doc in docs_rel_freq:
        for term, freq in doc.items():
            p_w_m[term] += freq / num_doc
    return p_w_m


def dirichlet_smoothing(tf, dl, c_tf, c_ctf):
    mu = 1500
    denom = tf + mu * (c_tf / c_ctf)
    nom = dl + mu
    return denom / nom


def div_by_doc_len(doc: Counter) -> Counter:
    doc_len = sum(doc.values())
    if doc_len == 0:
        return doc
    else:
        c = collections.Counter()
        for term, count in doc.items():
            c[term] = count / doc_len
        return c


def get_feature_weighted_model(claim_id,
                               perspective_id,
                               claim_text,
                               perspective_text,
                               collection_interface: CollectionInterface,
                               ):
    ranked_docs = collection_interface.get_ranked_documents_tf(claim_id, perspective_id)
    cp_tokens = nltk.word_tokenize(claim_text) + nltk.word_tokenize(perspective_text)
    cp_tokens = lmap(lambda x: x.lower(), cp_tokens)
    cp_tokens_count = collections.Counter(cp_tokens)

    def get_mention_prob(doc):
        dl = sum(doc.values())
        log_prob = 0

        for term, cnt in cp_tokens_count.items():
            c_tf, c_ctf = collection_interface.tf_collection(term)
            p_w = dirichlet_smoothing(cnt, dl, c_tf, c_ctf)
            log_prob += math.log(p_w)

        return math.exp(log_prob)

    docs_rel_freq = lmap(div_by_doc_len, ranked_docs)
    p_M_bar_D_list = lmap(get_mention_prob, docs_rel_freq)

    def apply_weight(e):
        doc_pre_freq, prob_M_bar_D = e
        return dict_value_map(lambda x: x * prob_M_bar_D, doc_pre_freq)

    weighted_doc_tf = lmap(apply_weight, zip(docs_rel_freq, p_M_bar_D_list))

    num_doc = len(weighted_doc_tf)
    p_w_m = average_tf_over_docs(weighted_doc_tf, num_doc)

    return p_w_m, num_doc


def re_tokenize(tokens):
    out = []
    for term in tokens:
        spliter = "-"
        if spliter in term and term.find(spliter) > 0:
            out.extend(term.split(spliter))
        else:
            out.append(term)
    return set(out)


def build_binary_feature(ci: DynRankedListInterface,
                         datapoint_list: List[PerspectiveCandidate]
                         ) -> List[Dict]:
    not_found_set = set()
    print("Load term stat")
    _, clue12_13_df = load_clueweb12_B13_termstat()
    cdf = 50 * 1000 * 1000

    def idf_scorer(doc: Counter, claim_text: str, perspective_text: str) -> bool:
        cp_tokens = nltk.word_tokenize(claim_text) + nltk.word_tokenize(perspective_text)
        cp_tokens = lmap(lambda x: x.lower(), cp_tokens)
        cp_tokens = set(cp_tokens)
        mentioned_terms = lfilter(lambda x: x in doc, cp_tokens)
        mentioned_terms = re_tokenize(mentioned_terms)

        def idf(term: str):
            if term not in clue12_13_df:
                if term in string.printable:
                    return 0
                not_found_set.add(term)

            return math.log((cdf+0.5)/(clue12_13_df[term]+0.5))

        score = sum(lmap(idf, mentioned_terms))
        max_score = sum(lmap(idf, cp_tokens))
        return score > max_score * 0.8

    def data_point_to_feature(x: PerspectiveCandidate) -> Dict:
        e = get_feature_binary_model(x.cid, x.pid, x.claim_text, x.p_text,
                                     ci, idf_scorer)
        feature: Counter = e[0]
        num_metion: int = e[1]
        return {
            'feature': feature,
            'cid': x.cid,
            'pid': x.pid,
            'num_mention': num_metion,
            'label': x.label
            }

    r = lmap(data_point_to_feature, datapoint_list)
    return r


def build_weighted_feature(datapoint_list):
    ci = CollectionInterface()

    def data_point_to_feature(data_point):
        label, cid, pid, claim_text, p_text = data_point
        return get_feature_weighted_model(cid, pid, claim_text, p_text, ci), label

    return lmap_w_exception(data_point_to_feature, datapoint_list, KeyError)


def get_doc_id(x):
    try:
        return x.doc_id
    except AttributeError:
        return x[0]

def get_feature_binary_model(claim_id,
                             perspective_id,
                             claim_text,
                             perspective_text,
                             ci: DynRankedListInterface,
                             is_mention_fn: Callable[[Counter[str], str, str], bool],
                             ) -> Tuple[Counter, int]:

    def is_mention(doc: Counter) -> bool:
        return is_mention_fn(doc, claim_text, perspective_text)

    print(claim_id, perspective_id)
    ranked_docs: List[GalagoDocRankEntry] = ci.query(claim_id, perspective_id, claim_text, perspective_text)
    ranked_docs = ranked_docs[:100]
    print("{} docs in ranked list".format(len(ranked_docs)))

    doc_id_list: List[str] = lmap(get_doc_id, ranked_docs)

    tf_d = load_multiple(CluewebDocTF, doc_id_list, True)
    not_found = []
    for idx, doc_id in enumerate(doc_id_list):
        if doc_id not in tf_d:
            not_found.append(idx)

    ranked_docs_tf = tf_d.values()
    mentioned_docs: List[Counter] = lfilter(is_mention, ranked_docs_tf)
    print("Found doc", len(tf_d), "mentioned doc", len(mentioned_docs))

    docs_rel_freq: List[Counter] = lmap(div_by_doc_len, mentioned_docs)
    num_doc: int = len(docs_rel_freq)
    p_w_m: Counter = average_tf_over_docs(docs_rel_freq, num_doc)

    return p_w_m, num_doc