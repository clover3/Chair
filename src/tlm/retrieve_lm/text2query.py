import nltk


def sent2query_simple(sent):
    tokens = nltk.wordpunct_tokenize(sent)
    return tokens


def doc2query_simple(cleaned_doc):
    tokens = nltk.wordpunct_tokenize(cleaned_doc)
    return tokens


def doc2query_textrank(cleaned_doc):
    return NotImplemented


