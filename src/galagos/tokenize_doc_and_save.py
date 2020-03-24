from collections import Counter

from nltk import tokenize

from datastore.table_names import BertTokenizedCluewebDoc, TokenizedCluewebDoc, CluewebDocTF


def tokenize_doc_and_save(buffered_saver, doc_id, core_text, tokenize_fn):
    # tokenize the docs
    chunks = bert_tokenize(core_text, tokenize_fn)
    buffered_saver.save(BertTokenizedCluewebDoc, doc_id, chunks)
    tokens = tokenize.word_tokenize(core_text)
    buffered_saver.save(TokenizedCluewebDoc, doc_id, tokens)
    tf = Counter(tokens)
    buffered_saver.save(CluewebDocTF, doc_id, tf)


def bert_tokenize(core_text, tokenize_fn):
    chunks = []
    for sent in tokenize.sent_tokenize(core_text):
        tokens = tokenize_fn(sent)
        chunks.append(tokens)
    return chunks