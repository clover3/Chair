import os
from typing import List

from arg.perspectives.basic_analysis import load_train_data_point, PerspectiveCandidate
from arg.perspectives.build_feature import get_doc_id
from arg.perspectives.ranked_list_interface import DynRankedListInterface, Q_CONFIG_ID_BM25_10000, make_doc_query
from cpath import output_path
from galagos.types import GalagoDocRankEntry
from list_lib import lmap, flatten


def main():
    ci = DynRankedListInterface(make_doc_query, Q_CONFIG_ID_BM25_10000)

    all_data_points = load_train_data_point()

    print("data_poing len" , len(all_data_points))
    def data_point_to_doc_id_list(x: PerspectiveCandidate) -> List[str]:
        ranked_docs: List[GalagoDocRankEntry] = ci.query(x.cid, x.pid, x.claim_text, x.p_text)
        ranked_docs = ranked_docs[:100]
        doc_id_list: List[str] = lmap(get_doc_id, ranked_docs)
        return doc_id_list

    doc_ids_list = lmap(data_point_to_doc_id_list, all_data_points)
    doc_ids = list(set(flatten(doc_ids_list)))
    print(len(doc_ids))

    save_path = os.path.join(output_path, "q_res_9_100")

    f = open(save_path, "w")
    for doc_id in doc_ids:
        f.write("{}\n".format(doc_id))
    f.close()


if __name__ == "__main__":
    main()
