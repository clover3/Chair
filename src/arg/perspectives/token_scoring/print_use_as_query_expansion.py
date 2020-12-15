import json
from typing import Dict

from arg.perspectives.load import claims_to_dict, get_all_claims
from arg.perspectives.pc_tokenizer import PCTokenizer
from exec_lib import run_func_with_config
from models.classic.stopword import load_stopwords_for_query


def main(config):
    word_list_path = config['word_list_path']
    claims = get_all_claims()
    claim_d = claims_to_dict(claims)
    stopwords = load_stopwords_for_query()

    word_list_d: Dict = json.load(open(word_list_path, "r"))


    tokenizer = PCTokenizer()

    for query_id in word_list_d:
        claim = claim_d[int(query_id)]
        word_list = word_list_d[query_id]
        base_query_terms = tokenizer.tokenize_stem(claim)
        base_query_terms = list([t for t in base_query_terms if t not in stopwords])
        #print

        new_term_set = set()
        for new_term in word_list:
            t = tokenizer.stemmer.stem(new_term)
            if t not in base_query_terms:
                new_term_set.add(t)

        print()
        print("Claim {}: {}".format(query_id, claim))
        print("base query terms: ", base_query_terms)
        print("new terms: ", new_term_set)


if __name__ == "__main__":
    run_func_with_config(main)
