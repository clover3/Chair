import json
import statistics
import sys
from collections import defaultdict

from krovetzstemmer import Stemmer

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from misc_lib import average, get_second
from tab_print import print_table
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TokenizedText, get_term_rep


def load_jsonl(input_path) -> list:
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.rstrip('\n|\r'))




def print_non_em(score_d):
    stemmer = Stemmer()
    def is_em_based(q_terms, d_terms):
        d_tokens_st = lmap(stemmer.stem, d_terms.split())
        q_tokens_st = lmap(stemmer.stem, q_terms.split())
        for token in q_tokens_st:
            if token in d_tokens_st:
                return True
        return False

    entries = []
    for term_pair, scores in score_d.items():
        if is_em_based(*term_pair):
            pass
        else:
            avg_score = average(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else "nan"
            entries.append((term_pair, avg_score, len(scores) / 5, std))
    entries.sort(key=lambda x: x[1], reverse=True)
    print_table(entries)


def print_gain_terms(score_d):
    entries = []
    for term_pair, scores in score_d.items():
        q_term, d_term = term_pair
        d_tokens = d_term.split()
        q_tokens = q_term.split()

        if len(q_tokens) > 1:
            if any([q_token in d_tokens for q_token in q_tokens]):
                q_term_part = " ".join([q_token for q_token in q_tokens if q_token in d_tokens])
                if (q_term_part, d_term) not in score_d:
                    continue
                q_part_score = average(score_d[q_term_part, d_term])
                q_double_score = average(score_d[q_term, d_term])
                gain = q_double_score - q_part_score
                entries.append((q_term, q_term_part, "/", d_term, q_double_score, gain))

        if len(d_tokens) > 1:
            if any([d_token in q_tokens for d_token in d_tokens]):
                d_term_part = " ".join([d_token for d_token in d_tokens if d_token in q_tokens])
                if (q_term, d_term_part) not in score_d:
                    continue
                part_score = average(score_d[q_term, d_term_part])
                double_score = average(score_d[q_term, d_term])
                gain = double_score - part_score
                entries.append((q_term, "/", d_term, d_term_part, double_score, gain))

    entries.sort(key=lambda x: x[4], reverse=True)
    print_table(entries)


def load_segment_log(jsonl):
    tokenizer = get_tokenizer()
    score_d = defaultdict(list)
    for j in jsonl:
        if "query" in j:
            query = j['query']
            doc = j['document']
            q_rep: TokenizedText = TokenizedText.from_text(tokenizer, query)
            d_rep: TokenizedText = TokenizedText.from_text(tokenizer, doc)
        else:
            q_indices = j['q_indices']
            d_indices = j['d_indices']
            score = j["score"]
            q_term = get_term_rep(q_rep, q_indices)
            d_term = get_term_rep(d_rep, d_indices)
            score_d[q_term, d_term].append(score)
    return score_d


def main():
    jsonl = load_jsonl(sys.argv[1])
    score_d = load_segment_log(jsonl)
    print_non_em(score_d)


if __name__ == "__main__":
    main()
