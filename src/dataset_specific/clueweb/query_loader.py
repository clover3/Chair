from typing import NamedTuple, Iterable, List
from xml.etree import ElementTree as ElementTree

from cpath import at_data_dir

all_years = range(2009, 2015)

def load_xml(file_path):
    content = open(file_path, "r").read()
    root = ElementTree.fromstring(content)
    return root


class TrecQuery(NamedTuple):
    query_id: str
    query_type: str
    keyword_query: str
    desc_query: str


def load_queries(year_list: Iterable[int]) -> List[TrecQuery]:
    all_queries = []
    for year in year_list:
        query_path = at_data_dir("clueweb", "{}.topics.xml".format(year))
        xml = load_xml(query_path)
        root_tag = xml.tag
        assert str(year) in root_tag
        for idx, topic in enumerate(xml):
            qid = topic.attrib['number']
            query_type = topic.attrib['type']
            keyword_query = topic.find('query').text
            desc_query = topic.find('description').text
            query = TrecQuery(qid, query_type, keyword_query, desc_query)
            all_queries.append(query)

    return all_queries
