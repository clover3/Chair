import os
import xml.etree.ElementTree as ET
from typing import List
from typing import NamedTuple

from contradiction.medical_claims.load_corpus import load_all_pmids
from cpath import data_path

doc_save_dir = os.path.join(data_path, "med_contradiction", "docs")


class AbstractText(NamedTuple):
    label: str
    nlm_category: str
    text: str


class Abstract(NamedTuple):
    text_list: List[AbstractText]


def load_xml(pmid):
    save_path = os.path.join(doc_save_dir, "{}.xml".format(pmid))
    return ET.parse(save_path)


def load_doc_parsed(pmid) -> Abstract:
    attrib_not_found = False
    def find(elem, tag):
        for first in elem.iter(tag):
            return first
        return None

    def get_attrib(node, name):
        if name in node.attrib:
            return node.attrib[name]
        nonlocal attrib_not_found
        attrib_not_found = True
        return None

    tree = load_xml(pmid)
    abstract_node = find(tree.getroot(), "Abstract")
    text_list = []
    if abstract_node:
        for node in abstract_node.findall("AbstractText"):
            category = get_attrib(node, "NlmCategory")
            label = get_attrib(node, "Label")
            text = node.text
            text_list.append(AbstractText(label, category, text))
    else:
        pass
        # print("Abstract not found", pmid)

    return Abstract(text_list)


def load_all_docs_parsed() -> List[Abstract]:
    d_list = []
    for pmid in load_all_pmids():
        d = load_doc_parsed(pmid)
        d_list.append(d)
    return d_list


if __name__ == "__main__":
    docs = load_all_docs_parsed()
    print("{} docs parsed".format(len(docs)))
