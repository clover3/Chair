import copy
import pickle

import nltk

from arg.claim_building.extract_sentences_from_ngram import get_compression_input_path
from cie.msc import takahe
from misc_lib import lmap
from models.classic.stopword import load_stopwords


def prac():
    ngram = ('pro', '-', 'life')
    ngram = ('partial', '-', 'birth')
    sents = pickle.load(open(get_compression_input_path(ngram), "rb"))

    st = 100
    ed = 1000
    sents = sents[st:ed]

    sents = list(set([" ".join(sent) for sent in sents]))
    sents = [s.split() for s in sents]
    for s in sents:
        print(s)

    def transform(text):
        text = [t for t in text if t]
        r = []
        for w, p in nltk.pos_tag(text):
            w = w.replace("/", "")
            r.append(w + "/" + p)
        return " ".join(r)

    print("pos tagging")
    tagged_community = lmap(transform, sents) # input sentences,  List[LIST[POS,WORD]]
    domain = 'meeting'  # meeting
    dataset_id = 'ami'  # ami, icsi
    language = 'en'  # en, fr
    development_or_test = 'test'  # development / test
    system_name = 'filippova'
    punct_tag = 'PUNCT'
    pos_separator = '/'
    nb_words = 12
    print("building graph")
    compresser = takahe.word_graph(
        system_name=system_name,
        tagged_community=copy.copy(tagged_community),
        language=language,
        punct_tag=punct_tag,
        pos_separator=pos_separator,
        lm=None,
        wv=None,
        stopwords=load_stopwords(),
        meeting_idf_dict=None,

        remove_stopwords=False,
        pos_filtering=True,
        stemming=True,
        cr_w=None,
        cr_weighted=None,
        cr_overspanning=None,
        nb_words=nb_words,
        diversity_n_clusters=None,
        keyphrase_reranker_window_size=0,
        common_hyp_threshold_verb=0.9,
        common_hyp_threshold_nonverb=0.3
    )

    # Write the word graph in the dot format
    # compresser.write_dot('new.dot')
    loose_verb_constraint = False
    summary = []
    while True:
        # Get the 200 best paths
        print("Starting compression")
        candidates = compresser.get_compression(nb_candidates=200, loose_verb_constraint=loose_verb_constraint)
        if len(candidates) > 0:
            final_paths = compresser.final_score(candidates, 1)  # n_results
            summary.append(final_paths[0][1])
            break
        # Then reason of no candidate:
        # 1. minimum number of words allowed in the compression larger than
        # the maximum path length in graph, then decrease nb_words and diversity_n_clusters
        else:
            compresser.nb_words -= 1
            if compresser.nb_words == 0:
                # 2. path should contain at least one verb, but no verb presented in the community
                # in this case, then loose the verb constraint
                loose_verb_constraint = True
                # raise RuntimeError("MSC failed")

    print(summary)


if __name__ == "__main__":
    prac()