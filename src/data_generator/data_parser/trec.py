import xml.sax
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from path import *
import math
import os
from cache import *
from misc_lib import TimeEstimator
from misc_lib import tprint
#from adhoc.bm25 import stem_tokenize

corpus_dir = os.path.join(data_path, "adhoc")
trecText_path = os.path.join(corpus_dir, "trecText")

class StreamHandler(xml.sax.handler.ContentHandler):

    lastEntry = None
    lastName  = None
    count = 0

    def startElement(self, name, attrs):
        self.lastName = name
        if name == 'DOC':
            self.lastEntry = {}
        elif name != 'root':
            self.lastEntry[name] = {'attrs': attrs, 'content': ''}

    def endElement(self, name):
        if name == 'DOC':
            print({
            'DOCNO' : self.lastEntry['DOCNO']['content'],
            'TEXT' : self.lastEntry['TEXT']['content'][:30]
            })
            self.count += 0
            if self.count > 10:
                raise StopIteration
            self.lastEntry = None
        elif name == 'root':
            raise StopIteration

    def characters(self, content):
        if self.lastEntry:
            self.lastEntry[self.lastName]['content'] += content


class TrecParser:
    def __init__(self, end_doc, tag_in_text = False):
        self.tags= []
        self.lastEntry = None
        self.end_doc = end_doc

    def feed(self, text):
        if text.startswith("<"):
            tag = text.strip()
            assert len(tag) < 20
            assert tag[-1] == ">"
            if tag[1] is not "/":
                self.tags.append(tag)
                if tag == "<DOC>":
                    self.lastEntry = {}
                    self.lastEntry["TEXT"] = ""
            else:
                self.tags.pop()
                if tag == "</DOC>":
                    self.end_doc(self.lastEntry)
        else: # Non Tag lines
            if self.tags[-1] == "<DOCNO>":
                self.lastEntry["DOCNO"] = text.strip()
            elif self.tags[-1] == "<TEXT>":
                self.lastEntry["TEXT"] += text



class TrecParser2:
    def __init__(self, end_doc):
        self.lastEntry = None
        self.end_doc = end_doc
        self.state = 2

    def feed(self, text):
        STATE_ROOT = 2
        STATE_DOC = 0
        STATE_TEXT = 1

        if self.state == STATE_ROOT:
            tag = text.strip()
            if tag == "<DOC>":
                self.state = STATE_DOC
                self.lastEntry = {}
                self.lastEntry["TEXT"] = ""
            else:
                assert False
        elif self.state == STATE_DOC:
            tag = text.strip()
            if tag.startswith("<DOCNO>"):
                st = len("<DOCNO>")
                ed = len(tag) - len("</DOCNO>")
                assert tag[ed:] == "</DOCNO>"
                self.lastEntry["DOCNO"] = tag[st:ed]
            elif tag == "<TEXT>":
                self.state = STATE_TEXT
            elif tag == "</DOC>":
                self.state = STATE_ROOT
                self.end_doc(self.lastEntry)
            else:
                print(tag)
                assert False
        if self.state == STATE_TEXT:
            end_tag = "</TEXT>\n"
            if text.endswith(end_tag):
                text = text[:-len(end_tag)]
                self.lastEntry["TEXT"] += text
                self.state = STATE_DOC


class TrecParser3:
    def __init__(self, end_doc):
        self.lastEntry = None
        self.text_arr = []
        self.end_doc = end_doc
        self.state = 2
        self.n_meta = 0
        self.line = 0
        self.end_tag = ""



    def feed(self, text):
        STATE_ROOT = 2
        STATE_DOC = 0
        STATE_CONTENT = 1

        def is_start_known_tag(text):
            text = text.strip()
            if text and text[0] == "<":
                if text.startswith("<DOCNO>"):
                    return "<DOCNO>"
                if text.startswith("<TEXT>"):
                    return "<TEXT>"
                if text.startswith("<HEADLINE>"):
                    return "<HEADLINE>"
                if text.startswith("<DATE>"):
                    return "<DATE>"
                if text.startswith("</DOC>"):
                    return "</DOC>"
            return False

        self.line += 1
        if self.state == STATE_ROOT:
            tag = text.strip()
            if tag == "<DOC>":
                self.state = STATE_DOC
                self.lastEntry = {}
                self.lastEntry["TEXT"] = ""
            elif len(tag) == 0:
                None

        elif self.state == STATE_DOC:
            tag = is_start_known_tag(text)
            if tag:
                text = text.strip()[len(tag):]
                if tag == "<DOCNO>":
                    ed = len(text) - len("</DOCNO>")
                    assert text[ed:] == "</DOCNO>"
                    self.lastEntry["DOCNO"] = text[:ed].strip()
                elif tag in ["<TEXT>", "<HEADLINE>", "<DATE>"]:
                    self.state = STATE_CONTENT
                    self.end_tag = tag[:1] + "/" + tag[1:]
                elif tag == "</DOC>":
                    self.state = STATE_ROOT
                    self.lastEntry["TEXT"] = "".join(self.text_arr)
                    self.end_doc(self.lastEntry)
                    text_len = len(self.lastEntry["TEXT"])
                    self.n_meta = 0
                    self.text_arr = []
                else:
                    self.n_meta += 1

        if self.state == STATE_CONTENT:
            end_tag = self.end_tag
            if text.strip().endswith(end_tag):
                text = text.strip()[:-len(end_tag)]
                #self.lastEntry["TEXT"] += text
                self.text_arr.append(text)
                self.state = STATE_DOC
            else:
                self.text_arr.append(text)
                #self.lastEntry["TEXT"] += text



def load_mobile_queries():
    query_path = os.path.join(corpus_dir, "test_query")

    buffer = open(query_path, "r").read()
    buffer = "<root>" + buffer + "</root>"
    root = ET.fromstring(buffer)

    for top in root:
        query_id = top[0].text
        query = top[1].text
        yield query_id, query


def get_inverted_index(collection):
    result = defaultdict(list)
    for doc_id, doc in collection.items():
        for word_pos, word in enumerate(doc.split()):
            result[word].append((doc_id, word_pos))

    return result

def get_tf_index(inv_index, lower=False):
    result = dict()
    for term, posting_list in inv_index.items():
        count_list = Counter()
        for doc_id, word_pos in posting_list:
            count_list[doc_id] += 1
        if lower:
            term = term.lower()
            if term not in result:
                result[term] = count_list
            else:
                for key in count_list.keys():
                    result[term][key] += count_list[key]
        else:
            result[term] = count_list


    return result


def collection_tf(docs):
    ctf = Counter()
    for doc in docs:
        tokens = stem_tokenize(doc)
        for t in tokens:
            ctf[t] += 1
    cl = sum(ctf.values())
    for key in ctf:
        ctf[key] = ctf[key] / cl

    return ctf



class Idf:
    def __init__(self, docs):
        self.df = Counter()
        self.idf = dict()
        for doc in docs:
            term_count = Counter()
            for token in doc.split():
                term_count[token] = 1
            for elem, cnt in term_count.items():
                self.df[elem] += 1
        N = len(docs)

        for term, df in self.df.items():
            self.idf[term] = math.log(N/df)
        self.default_idf = math.log(N/1)

    def __getitem__(self, term):
        if term in self.idf:
            return self.idf[term]
        else:
            return self.default_idf


def load_trec_data_proc():
    use_pickle = True
    if not use_pickle:
        print("loading collection...")
        collection = load_trec(trecText_path)
        idf = Idf(collection.values())

        print("building inverted index...")
        inv_index = get_inverted_index(collection)
        tf_index = get_tf_index(inv_index)
        queries = list(load_mobile_queries())

        save_data = (collection, idf, inv_index, tf_index, queries)
        save_to_pickle(save_data, "trecTask_all_info")
    else:
        collection, idf, inv_index, tf_index, queries = load_from_pickle("trecTask_all_info")

    return collection, idf, inv_index, tf_index, queries

def load_trec(path, dialect = 0):
    # use default ``xml.sax.expatreader``
    all_entry = []
    def callback(entry):
        all_entry.append(entry)

    if dialect == 0:
        parser = TrecParser(callback)
    elif dialect == 1:
        parser = TrecParser2(callback)
    elif dialect == 2:
        parser = TrecParser3(callback)

    with open(path, encoding='utf-8', errors='ignore') as f:
        for buffer in f:
            try:
                parser.feed(buffer)
            except StopIteration:
                break

    index_corpus = {}
    for entry in all_entry:
        index_corpus[entry["DOCNO"]] = entry["TEXT"]

    return index_corpus

    # if you can provide a file-like object it's as simple as


def load_robust(docs_dir, only_one_seg = False):
    collections = dict()
    for (dirpath, dirnames, filenames) in os.walk(docs_dir):
        for name in filenames:
            filepath = os.path.join(dirpath, name)
            tprint(filepath)
            d = load_trec(filepath, 2)
            print(len(d))
            collections.update(d)
            if only_one_seg:
                break
    return collections

from config.input_path import robust_path

if __name__ == '__main__':
    load_robust("/mnt/scratch/youngwookim/data/robust04")

