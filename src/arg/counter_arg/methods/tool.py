from collections import Counter

from nltk import sent_tokenize

from list_lib import flatten


def get_term_importance(bm25_module, sents):
    tokens = flatten([bm25_module.tokenizer.tokenize_stem(s) for s in sents])

    q_tf = Counter(tokens)
    term_importance = Counter()
    for term, tf in q_tf.items():
        term_importance[term] += bm25_module.term_idf_factor(term) * tf
    return term_importance


def sent_tokenize_newline(text):
    sents = sent_tokenize(text)
    r = []
    for s in sents:
        for new_sent in s.split("\n"):
            r.append(new_sent)
    return r