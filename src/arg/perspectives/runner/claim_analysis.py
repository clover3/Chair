from collections import Counter
from typing import List

import spacy

from arg.perspectives.load import get_claims_from_ids, load_train_claim_ids


def featurize(nlp, text, param):
    k1 = param['k1']
    doc = nlp(text)

    tf = Counter()

    tokens_by_loc = []
    for token in doc:
        tokens_by_loc.append({})
        tf[token.text] = 1

    k1_set = set()
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            k1_set.add(doc[i].text)
    for term in k1_set:
        tf[term] += k1
    return tf


def run_get_claim_term_weighting():
    # Load claim
    # Do dependency parsing
    # show top level roles ( dobj, sub )
    param = {'k1': 0.5}
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    out_d = get_claim_term_weighting(claims, param)
    nlp = spacy.load("en_core_web_sm")

    for c in claims:
        weight = out_d[c['cId']]
        for token in nlp(c['text']):
            s = token.text
            if weight[s] > 1:
                s = "[{}]".format(s)
            print(s, end=" ")
        print()
        # print(c['text'])
        # print(out_d[c['cId']])


def get_claim_term_weighting(claims, param):
    nlp = spacy.load("en_core_web_sm")
    out_d = {}
    for c in claims:
        text = c['text']
        tf = featurize(nlp, text, param)
        out_d[c['cId']] = tf
    return out_d





def init_spacy():
    print(spacy.explain('expl'))
    nlp = spacy.load("en_core_web_sm")
    text1 = "There is still a place for mercenaries working for NGOs."
    text2 = "Humanitarian mercenaries"
    text3 = "Legislation against mercenaries"

    for text in [text1, text2, text3]:
        doc = nlp(text)
        print(doc)
        for token in doc:
            print(token.text, token.dep_, token.head.text)
        print("--------")

    for chunk in doc.noun_chunks:
        print(chunk.text, chunk.root.text, chunk.root.dep_,
            chunk.root.head.text)
        print(chunk.start, chunk.end)
        #print(dir(chunk))


def dependency_simple():
    nlp = spacy.load("en_core_web_sm")
    text = "There is still a place for mercenaries working for NGOs."
    doc = nlp(text)
    print(doc)
    for idx, token in enumerate(doc):

        print(token.text, token.dep_, token.head.text)
    print("--------")


from nltk.parse.corenlp import CoreNLPParser


def has_sent(parser, text):
    parse = next(parser.raw_parse(text))
    has_sent = False
    for item in parse.subtrees():
        if item.label() == "S":
            has_sent = True
    return has_sent


def stanford_nlp():
    parser = CoreNLPParser()
    text1 = "There is still a place for mercenaries working for NGOs."
    text2 = "The Rich Poor Gap Silences the Political Voice of the Poor"
    text3 = "Legislation against mercenaries"
    for text in [text1, text2, text3]:
        parse = next(parser.raw_parse(text))
        print(parse)
        has_sent = False
        for item in parse.subtrees():
            if item.label() == "S":
                has_sent = True
        print(has_sent)


if __name__ == "__main__":
    init_spacy()