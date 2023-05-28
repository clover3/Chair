from collections import Counter

from krovetzstemmer import Stemmer
from transformers import AutoTokenizer
from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_measure_single_q_term.compute_gains_resourced import load_resources


def load_terms():
    term_path = path_join(output_path, "msmarco", "terms10K.txt")
    with open(term_path, 'r', encoding="utf-8") as file:
        for line in file:
            term = line.strip()
            yield term


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    terms_in_bert_voca = list(load_terms())
    doc_id_to_tf, _, _ = load_resources("when_full_re_0")

    stemmer = Stemmer()
    terms_stemmed = set()
    for t in terms_in_bert_voca:
        terms_stemmed.add(stemmer.stem(t))

    print("{} terms after stemming".format(len(terms_stemmed)))
    terms_not_seen = set(terms_stemmed)
    for doc_id, tf in doc_id_to_tf.items():
        for t in tf:
            if t in terms_not_seen:
                terms_not_seen.remove(t)


    terms_seen = terms_stemmed - terms_not_seen

    term_path = path_join(output_path, "msmarco", "terms10K_stemmed.txt")
    with open(term_path, "w") as file:
        for item in terms_seen:
            file.write("%s\n" % item)

    print("{} terms after filtering".format(len(terms_seen)))



if __name__ == "__main__":
    main()
