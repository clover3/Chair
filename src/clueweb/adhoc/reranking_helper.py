from typing import List, Iterable, Dict

from clueweb.adhoc.load_docs import read_doc_id_title_text
from cpath import at_data_dir, at_output_dir
from evals.parse import load_qrels_structured
from evals.types import QRelsDict
from trec.trec_parse import load_ranked_list_grouped, TrecRankedListEntry


def load_qrels_for(year) -> QRelsDict:
    qrel_path = at_data_dir("clueweb", "{}.qrels.txt".format(year))
    return load_qrels_structured(qrel_path)


def load_clueweb09_ranked_list() -> Dict[str, List[TrecRankedListEntry]]:
    rl_path = at_output_dir("clueweb", "clue09_ranked_list.txt")
    return load_ranked_list_grouped(rl_path)


class CluewebReranking:
    def __init__(self,
                 target_year_list: List[int]):
        self.target_year_list = target_year_list

        qrels = {}
        for year in self.target_year_list:
            qrels.update(load_qrels_for(year))
        self.qrels = qrels
        self.clueweb09_ranked_list = load_clueweb09_ranked_list()

    def get_all_queries_sorted(self):
        query_ids = set()
        query_ids.update(self.qrels.keys())
        query_ids.update(self.clueweb09_ranked_list.keys())
        query_ids_list = list(query_ids)
        query_ids_list.sort()
        return query_ids_list

    def get_docs_for_training(self, query_id) -> Iterable[str]:
        qrel_entries: Dict = self.qrels[query_id]
        docs_from_candidates = map(TrecRankedListEntry.get_doc_id, self.clueweb09_ranked_list[query_id])

        docs = set()
        docs.update(qrel_entries.keys())
        docs.update(docs_from_candidates)
        return docs

    def get_docs_for_testing(self, query_id) -> Iterable[str]:
        docs_from_candidates = map(TrecRankedListEntry.get_doc_id, self.clueweb09_ranked_list[query_id])
        return docs_from_candidates


def main():
    cr = CluewebReranking(list(range(2010, 2013)))
    all_docs = read_doc_id_title_text()

    f = open(at_output_dir("clueweb", "not_found.txt"), "w")
    not_found = 0
    for qid in cr.qrels.keys():
        for doc_id in cr.get_docs_for_training(qid):
            if doc_id not in all_docs:
                not_found += 1
                f.write("{}\n".format(doc_id))

    print("not found", not_found)

##

if __name__ == "__main__":
    main()