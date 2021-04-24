import os
import xml.etree.ElementTree as ET
from typing import List, NamedTuple

from cpath import data_path


class Claim(NamedTuple):
    pmid: str
    assertion: str
    question: str
    type: str
    text: str


class Review(NamedTuple):
    pmid: str
    title: str
    claim_list: List[Claim]


def get_corpus_path():
    corpus_path = os.path.join(data_path, "med_contradiction", "corpus.xml")
    return corpus_path


def get_corpus_xml():
    tree = ET.parse(get_corpus_path())
    return tree


def load_all_pmids() -> List[str]:
    tree = get_corpus_xml()

    pmid_list = []
    for review in tree.findall("REVIEW"):
        review_pmid = review.attrib['REVIEW_PMID']
        pmid_list.append(review_pmid)

        for claim in review.findall("CLAIM"):
            claim_pmid = claim.attrib['PMID']
            pmid_list.append(claim_pmid)
    return list(set(pmid_list))


def load_parsed() -> List[Review]:
    tree = get_corpus_xml()
    review_list: List[Review] = []
    for review_elem in tree.findall("REVIEW"):
        review_pmid = review_elem.attrib['REVIEW_PMID']
        review_title = review_elem.attrib['REVIEW_TITLE']

        claim_list = []
        for claim_elem in review_elem.findall("CLAIM"):
            pmid = claim_elem.attrib['PMID']
            assertion = claim_elem.attrib['ASSERTION']
            question = claim_elem.attrib['QUESTION']
            claim_type = claim_elem.attrib['TYPE']
            text = claim_elem.text
            claim = Claim(pmid, assertion, question, claim_type, text)
            claim_list.append(claim)
        review = Review(review_pmid, review_title, claim_list)
        review_list.append(review)

    return review_list


def load_all_claims() -> List[Claim]:
    output = []
    for review in load_parsed():
        output.extend(review.claim_list)
    return output


if __name__ == "__main__":
    review_list = load_parsed()
    print(review_list[0])
    print(review_list[0].claim_list[0])

