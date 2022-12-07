from typing import List

from adhoc.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_all_claim_d
from clueweb.sydney_path import index_list
from galagos.interface import format_query_bm25
from galagos.tokenize_util import TokenizerForGalago
from list_lib import lmap


def generate_query(qid_list: List[str]):
    million = 1000 * 1000
    clueweb_tf, clueweb_df = load_clueweb12_B13_termstat()
    claim_d = get_all_claim_d()
    tokenizer = TokenizerForGalago()
    def get_query_for_qid(qid: str):
        cid_s, pid_s = qid.split("_")
        claim_text = claim_d[int(cid_s)]
        perspective_text = perspective_getter(int(pid_s))
        query_text = claim_text + " " + perspective_text
        query_tokens = tokenizer.tokenize(query_text)

        candidate_terms = []
        for term in query_tokens:
            if clueweb_df[term] > million:
                pass
            else:
                candidate_terms.append(term)

        print(candidate_terms)
        return format_query_bm25(qid, candidate_terms)

    return lmap(get_query_for_qid, qid_list)


def make_galago_query(queries):
    return {
        "queries": queries,
        "index": index_list,
        "requested": 10000,
    }