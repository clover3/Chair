import json
import os
from typing import List, Dict, Counter

from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_perspectrum_golds, PerspectiveCluster, get_all_claim_d
from clueweb.sydney_path import index_list
from cpath import output_path
from galagos.interface import format_query_bm25
from galagos.tokenize_util import TokenizerForGalago
from list_lib import lmap
from models.classic.lm_util import tokens_to_freq, average_counters


def generate_query(target_claim_ids: List[int]):
    million = 1000 * 1000
    gold_d: Dict[int, List[PerspectiveCluster]] = load_perspectrum_golds()
    clueweb_tf, clueweb_df = load_clueweb12_B13_termstat()

    claim_d = get_all_claim_d()
    tokenizer = TokenizerForGalago()

    def get_cluster_lm(cluster: PerspectiveCluster) -> Counter:
        p_text_list: List[str] = lmap(perspective_getter, cluster.perspective_ids)
        tokens_list: List[List[str]] = lmap(tokenizer.tokenize, p_text_list)
        counter_list = lmap(tokens_to_freq, tokens_list)
        counter = average_counters(counter_list)
        return counter

    def get_query_for_cid(cid: int):
        claim_text = claim_d[cid]
        cluster_list = gold_d[cid]
        counter_list: List[Counter] = lmap(get_cluster_lm, cluster_list)
        counter: Counter = average_counters(counter_list)
        claim_tokens = tokenizer.tokenize(claim_text)

        candidate_terms = claim_tokens
        for term, tf in counter.most_common(10):
            if clueweb_df[term] > million :
                pass
            else:
                candidate_terms.append(term)

        print(candidate_terms)
        return format_query_bm25(str(cid), candidate_terms)

    return lmap(get_query_for_cid, target_claim_ids)


def make_galago_query(queries):
    return {
        "queries": queries,
        "index": index_list,
        "requested": 10000,
    }



def main():
    target_claim_ids = [47, 504]
    queries = generate_query(target_claim_ids)
    g_queries = make_galago_query(queries)
    out_path = os.path.join(output_path, "ca_building", "run3", "queries_2.json")
    fout = open(out_path, "w")
    fout.write(json.dumps(g_queries, indent=True))
    fout.close()


if __name__ == "__main__":
    main()