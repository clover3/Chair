import json
from collections import Counter

from transformers import AutoTokenizer

from cpath import output_path
from misc_lib import path_join, get_second
from trainer_v2.per_project.transparency.mmp.dev_analysis.when_term_frequency import enum_when_corpus

# Build global alignment

def when_raw_tf():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    term_tf = Counter()
    for query, doc_pos, doc_neg in enum_when_corpus():
        for doc in [doc_pos, doc_neg]:
            input_ids = tokenizer(doc)["input_ids"]
            for t in input_ids:
                term_tf[t] += 1
    return term_tf


def main():
    when_file = open(path_join(output_path, "msmarco", "when_tf"), "r")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # enumerate local alignment, average them
    term_tf = when_raw_tf()
    non_zero_tf = Counter()
    term_score_sum = Counter()
    for line in when_file:
        j = json.loads(line)
        logits = j['logits']
        aligns = j['aligns']
        if logits[0] > 0.5:
            for index, token_id, score in aligns:
                non_zero_tf[token_id] += 1
                term_score_sum[token_id] += score

    entries = []
    for t in non_zero_tf:
        n_all = term_tf[t]
        avg_align_score = term_score_sum[t] / n_all
        entries.append((t, avg_align_score))

    entries.sort(key=get_second, reverse=True)

    for token_id, avg_score in entries:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        row = [token_id, token, avg_score, term_tf[token_id], non_zero_tf[token_id]]
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()
