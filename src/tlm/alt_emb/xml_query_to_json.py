import xml.etree.ElementTree as ET
from typing import List, Dict

from nltk import word_tokenize

from galagos.parse import save_queries_to_file, clean_query, get_query_entry_bm25_anseri
from galagos.types import Query
from list_lib import lmap


def load_xml_query(xml_path) -> List[Query]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root.tag == "queries"

    query_list = []
    for child in root:
        assert child.tag == 'query'
        q_id = child.find('id').text
        text = child.find('title').text

        query = Query(qid=q_id, text=text)
        query_list.append(query)

    return query_list


def xml_query_to_json(xml_path, json_path):
    queries: List[Query] = load_xml_query(xml_path)

    def transform(q: Query) -> Dict:
        tokens = word_tokenize(q.text)
        tokens = clean_query(tokens)
        return get_query_entry_bm25_anseri(q.qid, tokens)

    queries_dict_list: List[Dict] = lmap(transform, queries)
    save_queries_to_file(queries_dict_list, json_path)




