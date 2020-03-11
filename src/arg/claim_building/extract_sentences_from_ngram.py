import os
import pickle

from nltk import ngrams

from arg.claim_building.count_ngram import merge_subword
from cpath import output_path
from misc_lib import exist_or_mkdir, assign_list_if_not_exists
from tlm.ukp.sydney_data import dev_pretend_ukp_load_tokens_for_topic


def extract_sentences_from_ngram(docs, target_n_gram):
    output = {}
    for doc in docs:
        for sent in doc:
            sent = merge_subword(sent)
            for ngram in ngrams(sent, 3):
                if ngram in target_n_gram:
                    assign_list_if_not_exists(output, ngram)
                    output[ngram].append(sent)
    return output


def get_compression_input_path(ngram):
    out_dir = os.path.join(output_path, "compression")
    file_name = "".join(ngram)
    out_path = os.path.join(out_dir, file_name)
    return out_path


def runner():
    ngrams = [('pro', '-', 'life'),
              ('pro', '-', 'choice'),
              ('partial', '-', 'birth')
              ]
    topic = "abortion"
    out_dir = os.path.join(output_path, "compression")
    token_dict = dev_pretend_ukp_load_tokens_for_topic(topic)
    docs = token_dict.values()
    exist_or_mkdir(out_dir)
    sents_d = extract_sentences_from_ngram(docs, ngrams)
    for ngram in ngrams:
        out_path = get_compression_input_path(ngram)
        if ngram in sents_d:
            sents = sents_d[ngram]
            pickle.dump(sents, open(out_path, "wb"))
        else:
            print("Warning {} not found".format(ngram))


if __name__ == "__main__":
    runner()