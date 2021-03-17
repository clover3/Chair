import pickle
from collections import Counter
from typing import List, Dict, Tuple
from typing import NamedTuple

from arg.bm25 import BM25Bare
from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat_stemmed, cdf
from arg.perspectives.pc_tokenizer import pc_tokenize_ex
from clueweb.html_to_text import get_text_from_html
from galagos.parse import load_query_json_as_dict
from list_lib import lmap
from misc_lib import find_max_idx, get_first
from models.classic.stemming import StemmedToken
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


class Doc(NamedTuple):
    doc_id: str
    content: str
    core_text: str


def get_best_segment(bm25_module: BM25Bare, html_content: str, query: str) -> Tuple[str, float]:
    text: str = get_text_from_html(html_content)
    query_tokens: List[StemmedToken] = pc_tokenize_ex(query)
    doc_tokens: List[StemmedToken] = pc_tokenize_ex(text)

    def get_tf(tokens: List[StemmedToken]) -> Counter:
        return Counter(lmap(StemmedToken.get_stemmed_token, tokens))

    q_tf = get_tf(query_tokens)
    idx = 0
    window_size = 50
    step = 30
    candidate_list = []
    while idx < len(doc_tokens):
        st = idx
        ed = idx + window_size
        t_tf = get_tf(doc_tokens[st:ed])
        score = bm25_module.score_inner(q_tf, t_tf)
        candidate = score, st, ed
        candidate_list.append(candidate)
        idx += step

    max_idx = find_max_idx(candidate_list, get_first)
    score, st, ed = candidate_list[max_idx]
    passage_text = " ".join(map(StemmedToken.get_raw_token, doc_tokens[st:ed]))
    return passage_text, score


def load_doc_pickle(path) -> Dict[str, Doc]:
    obj = pickle.load(open(path, "rb"))
    out_d = {}
    for d in obj:
        d_strct = Doc(d['doc_id'], d['html'], d['core_text'])
        out_d[d_strct.doc_id] = d_strct
    return out_d


def get_bm25_module() -> BM25Bare:
    tf, df = load_clueweb12_B13_termstat_stemmed()
    return BM25Bare(df, avdl=11.7, num_doc=cdf, k1=0.00001, k2=100, b=0.5)


def main():
    q_res_path = "/mnt/nfs/work3/youngwookim/data/counter-arg/q_res.txt"
    doc_pickle_path = "/mnt/nfs/work3/youngwookim/data/counter-arg/docs.pickle"
    query_json_path = "/mnt/nfs/work3/youngwookim/data/counter-arg/search_query.json"
    query_text_d: Dict[str, str] = load_query_json_as_dict(query_json_path)

    ranked_list_group: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(q_res_path)
    docs_dict: Dict[str, Doc] = load_doc_pickle(doc_pickle_path)

    bm25_module = get_bm25_module()
    # print
    # I want to get top candidate passages (docs)
    for query_id, ranked_list in ranked_list_group.items():
        query = query_text_d[query_id]
        e_list = []
        doc_id_set = set()
        for entry in ranked_list:
            assert entry.doc_id not in doc_id_set
            doc_id_set.add(entry.doc_id)
            doc = docs_dict[entry.doc_id]
            passage_text, score = get_best_segment(bm25_module, doc.content, query)
            e = entry.doc_id, passage_text, score
            e_list.append(e)

        e_list.sort(key=lambda x: x[2], reverse=True)

        print("Query: ", query)
        for doc_id, passage_text, score in e_list:
            print(doc_id)
            print(passage_text)
            print("{0:.2f}".format(score))


if __name__ == "__main__":
    main()
