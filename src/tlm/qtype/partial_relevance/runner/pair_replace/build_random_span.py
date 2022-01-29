import pickle
import random
from typing import Iterable

import spacy
from krovetzstemmer import Stemmer
from nltk import sent_tokenize
from nltk.util import ngrams

from cache import save_to_pickle, load_from_pickle
from cpath import at_output_dir
from dataset_specific.msmarco.common import MSMarcoDoc
from dataset_specific.msmarco.enum_documents import enum_documents
from explain.genex.idf_lime import load_idf_fn_for


def enum_document(skip_rate=10) -> Iterable[MSMarcoDoc]:
    for idx, doc in enumerate(enum_documents()):
        if idx % skip_rate > 0:
            continue
        yield doc


def get_syntactic_tag(tokens):
    covering_root = None
    for t in tokens:
        n_descendant = 0
        covered = True
        for descendant in t.subtree:
            if descendant not in tokens:
                covered = False
            n_descendant += 1
        if covered:
            if n_descendant + 1 == len(tokens):
                covering_root = t
                break

    if covering_root is None:
        return covering_root
    else:
        return covering_root.dep_


# TODO
#   given (qt,dt), find n-gram vocabulary that are 'comparable' to dt.
#
# Example dt spans
#   found in soil and dust?
#   are found in school?
#   Start with random documents.
#   Parse with SpaCy
#   entry = Tuple[Word, syntactic relation to parent]
#                           - None if they tokens does not match any subtree.


def build_ngrams():
    nlp = spacy.load("en_core_web_sm")
    def enum_ngrams(tokens):
        for n in range(1, 5):
            yield from ngrams(tokens, n)

    seen_ngram = set()
    def remove_duplicate_update_seen(ngram_list):
        def get_rep(ngram):
            return " ".join([str(t) for t in ngram])
        rep_list = map(get_rep, ngram_list)

        selected_outputs = []
        for idx, rep in enumerate(rep_list):
            if rep in seen_ngram:
                pass
            else:
                seen_ngram.add(rep)
                selected_outputs.append(ngram_list[idx])

        return selected_outputs

    all_entries = []
    for doc in enum_document():
        sents = [doc.title] + sent_tokenize(doc.body)
        sent, = random.sample(sents, 1)
        print("Sentence:", sent)
        parsed_sent = nlp(sent)
        n_gram_list = list(enum_ngrams(parsed_sent))
        n_gram_list = remove_duplicate_update_seen(n_gram_list)
        entries_per_document = []
        for ngram in n_gram_list:
            syntactic_tag = get_syntactic_tag(ngram)
            ngram_s = [str(t) for t in ngram]
            entry = (ngram_s, syntactic_tag)
            entries_per_document.append(entry)

        all_entries.append(entries_per_document)

        if len(all_entries) > 100:
            break
    save_to_pickle(all_entries, "msmarco_random_spans")


def augment_idf():
    all_entries = load_from_pickle("msmarco_random_spans")
    get_idf = load_idf_fn_for("tdlt")
    stemmer = Stemmer()

    idf_unknown = get_idf("Somewordthatdoesnotexists_indataset")
    assert idf_unknown > 16

    def is_unknown_score(s):
        return abs(idf_unknown-s) < 1e-6

    n_all = 0
    output = []
    for entries_per_doc in all_entries:
        new_entries_per_doc = augment_filter(entries_per_doc, get_idf, is_unknown_score, stemmer)
        output.append(new_entries_per_doc)
        n_all += len(new_entries_per_doc)
    print("{} spans saved".format(n_all))
    save_path = at_output_dir("qtype", "msmarco_random_spans.pickle")
    pickle.dump(output, open(save_path, "wb"))


def augment_filter(entries_per_doc, get_idf, is_unknown_score, stemmer):
    new_entries_per_doc = []
    for ngram_s, tag in entries_per_doc:
        idf_sum = 0
        reject = False
        for word in ngram_s:
            idf = get_idf(stemmer.stem(word))
            if is_unknown_score(idf):
                reject = True
            idf_sum += idf

        if not reject:
            new_entry = (ngram_s, tag, idf_sum)
            new_entries_per_doc.append(new_entry)
    return new_entries_per_doc


def main():
    print("Augment idf")
    augment_idf()


if __name__ == "__main__":
    main()