from typing import List

from cache import load_from_pickle
from datastore.interface import load
from datastore.table_names import QueryResult
from galagos.types import SimpleRankedListEntry
from list_lib import lmap
from sydney_clueweb.clue_path import get_first_disk


class RankedListInterface:
    def __init__(self):
        print("RankedListInterface __init__")
        self.disk_path = get_first_disk()
        self.collection_tf = load_from_pickle("collection_tf")
        self.num_request = 10

    def fetch_from_q_res_id(self,
              query_res_id: str,
              ) -> List[SimpleRankedListEntry]:
        def translate_structure(raw_data) -> List[SimpleRankedListEntry]:
            try:
                dummy = raw_data[0].doc_id
                r = raw_data
            except AttributeError:
                def tuple_to_ranked_entry(tuple) -> SimpleRankedListEntry:
                    doc_id, rank, score = tuple
                    return SimpleRankedListEntry(doc_id=doc_id,
                                                 rank=rank,
                                                 score=score)

                r = lmap(tuple_to_ranked_entry, raw_data)
            return r

        try:
            raw_data = load(QueryResult, query_res_id)
            data = translate_structure(raw_data)
            return data
        except KeyError:
            print(query_res_id)
            raise