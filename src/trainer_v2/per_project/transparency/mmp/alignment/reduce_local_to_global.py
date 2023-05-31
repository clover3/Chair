import json
from collections import Counter

from transformers import AutoTokenizer

from misc_lib import get_second


def avg_summarize_local_aligns(corpus_tf, input_file_path, save_path):
    when_file = open(input_file_path, "r")
    save_f = open(save_path, "w")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # enumerate local alignment, average them
    non_zero_tf = Counter()
    term_score_sum = Counter()
    j_itr = map(json.loads, when_file)
    for j in j_itr:
        logits = j['logits']
        aligns = j['aligns']
        if logits[0] > 0.5:
            for index, token_id, score in aligns:
                non_zero_tf[token_id] += 1
                term_score_sum[token_id] += score
    entries = []
    for t in non_zero_tf:
        n_all = corpus_tf[t]
        avg_align_score = term_score_sum[t] / n_all
        entries.append((t, avg_align_score))
    entries.sort(key=get_second, reverse=True)
    for token_id, avg_score in entries:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        row = [token_id, token, avg_score, corpus_tf[token_id], non_zero_tf[token_id]]
        out_line = "\t".join(map(str, row))
        save_f.write(out_line + "\n")


def sum_summarize_local_aligns_paired(corpus_tf, input_file_path, save_path):
    when_file = open(input_file_path, "r")
    save_f = open(save_path, "w")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # enumerate local alignment, average them
    non_zero_tf = Counter()
    term_score_sum = Counter()
    j_itr = map(json.loads, when_file)
    for j in j_itr:
        logits = j['logits']
        aligns = j['aligns']
        if logits[0] > 0.5:
            for q_term_id, d_term_id, score in aligns:
                non_zero_tf[q_term_id, d_term_id] += 1
                term_score_sum[q_term_id, d_term_id] += score
    entries = []
    for t in non_zero_tf:
        qt, dt = t
        reduced_score = term_score_sum[t] / corpus_tf[dt]
        entries.append((t, reduced_score))
    entries.sort(key=get_second, reverse=True)
    for pair, acc_score in entries:
        q_term_id, d_term_id = pair
        def conv_token_id(token_id):
            return tokenizer.convert_ids_to_tokens([token_id])[0]

        q_term = conv_token_id(q_term_id)
        d_term = conv_token_id(d_term_id)

        row = [q_term, d_term, acc_score,
               non_zero_tf[q_term_id, d_term_id],
               corpus_tf[q_term_id], corpus_tf[d_term_id]]
        out_line = "\t".join(map(str, row))
        save_f.write(out_line + "\n")
