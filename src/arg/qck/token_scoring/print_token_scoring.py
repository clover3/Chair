from typing import List, Dict

import math
import numpy as np

from arg.perspectives.load import load_claims_for_sub_split, claims_to_dict
from arg.qck.token_scoring.collect_score import WordAsID, decode_word_as_id
from cache import load_pickle_from
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from exec_lib import run_func_with_config
from misc_lib import get_second
from models.classic.stopword import load_stopwords_ex


def main(config):
    split = config['split']
    word_prob_path = config['word_prob_path']
    per_query_infos: Dict[str, Dict[WordAsID, np.array]] = load_pickle_from(word_prob_path)
    claims = load_claims_for_sub_split(split)
    claim_d = claims_to_dict(claims)
    stopwords = load_stopwords_ex()

    def is_stopword(tokens):
        if len(tokens) == 1 and tokens[0] in stopwords:
            return True
        else:
            return False

    tokenizer = get_tokenizer()

    for query_id, d in per_query_infos.items():
        entry = []
        for key in d.keys():
            tokens: List[str] = decode_word_as_id(tokenizer, key)
            if is_stopword(tokens):
                continue

            plain_word: str = pretty_tokens(tokens, True)
            pos, neg = d[key]
            pos_log = math.log(pos + 1e-10)
            neg_log = math.log(neg + 1e-10)
            diff = pos_log - neg_log
            entry.append((plain_word, diff, pos_log, neg_log))

        print(query_id, claim_d[int(query_id)])
        entry.sort(key=get_second, reverse=True)
        for word, diff, pos, neg in entry[:100]:
            word = word.strip()
            print("{0}\t{1:.2f}\t{2:.2f}\t{3:.2f}".format(word, diff, pos, neg))


if __name__ == "__main__":
    run_func_with_config(main)
