from typing import List, Dict

from arg.qck.decl import KnowledgeDocument, QCKQuery, QKUnit
from arg.qck.kd_candidate_gen import iterate_document_parts
from arg.robust.qc_common import to_qck_queries
from base_type import FilePath
from cache import load_from_pickle, save_to_pickle
from data_generator.data_parser.robust import load_robust04_title_query
from galagos.parse import load_galago_ranked_list_w_space
from galagos.types import GalagoDocRankEntry


def make_qk_unit_list(qck_queries: List[QCKQuery],
         ranked_list_dict: Dict[str, List[GalagoDocRankEntry]],
         doc_tokens: Dict[str, List[str]],
         window_size,
         step_size,
         max_per_doc
         ) -> List[QKUnit]:

    output: List[QKUnit] = []
    for q in qck_queries:
        ranked_list = ranked_list_dict[q.query_id]
        kd_list: List[KnowledgeDocument] = []
        for entry in ranked_list:
            tokens = doc_tokens[entry.doc_id]
            kd = KnowledgeDocument(entry.doc_id, tokens)
            kd_list.append(kd)

        kdp_list = iterate_document_parts(kd_list, window_size, step_size, max_per_doc)
        qk: QKUnit = (q, kdp_list)
        output.append(qk)

    return output


def load_doc_tokens():
    return load_from_pickle("robust_on_wiki_tokens")


def main():
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/qck/robust_on_wiki/q_res.txt")
    queries: Dict[str, str] = load_robust04_title_query()
    qck_queries = to_qck_queries(queries)
    ranked_list: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list_w_space(q_res_path)
    doc_tokens = load_doc_tokens()
    window_size = 256
    step_size = 256
    max_per_doc = 10
    qk_units: List[QKUnit] = make_qk_unit_list(qck_queries, ranked_list, doc_tokens, window_size, step_size, max_per_doc)
    save_to_pickle(qk_units, "robust_on_wiki_qk_candidate")


if __name__ == "__main__":
    main()