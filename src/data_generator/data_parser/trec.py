import xml.sax
import time


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
    def __init__(self, end_doc):
        NotImplementedError

        self.tags= []
        self.lastEntry = None
        self.end_doc = end_doc

    def feed(self, text):
        if text.startswith("<"):
            tag = text.strip()
            assert tag[-1] == ">"
            if tag[1] is not "/":
                self.tags.append(tag)
                if tag == "<DOC>":
                    self.lastEntry = {}
                    self.lastEntry["TEXT"] = ""
            else:
                self.tags = self.tags[:-1]
                if tag == "</DOC>":
                    self.end_doc(self.lastEntry)
        else:
            if self.tags[-1] == "<DOCNO>":
                self.lastEntry["DOCNO"] = text.strip()
            elif self.tags[-1] == "<TEXT>":
                self.lastEntry["TEXT"] += text




def load_trec(path):
    # use default ``xml.sax.expatreader``
    all_entry = []
    def callback(entry):
        all_entry.append(entry)

    parser = TrecParser(callback)
    with open(path, encoding='utf-8') as f:
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


if __name__ == '__main__':
    load_trec("../../../data/adhoc/trecText")
