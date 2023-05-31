import os.path
from collections import Counter

from transformers import AutoTokenizer

from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.alignment.reduce_local_to_global import avg_summarize_local_aligns
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


def get_attn_score_jsonl_path(input_file_name):
    input_file_path = path_join(
        output_path, "msmarco", "passage", "attn_scores", input_file_name)
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(input_file_path)
    return input_file_path


def get_avg_score_save_path(file_name):
    input_file_path = path_join(output_path, "msmarco", "passage", file_name)
    return input_file_path


def when_tf():
    input_file_name = "when_tf"
    input_file_path = get_attn_score_jsonl_path(input_file_name)
    save_name = "when_tf.avg_score_summary"
    save_path = get_avg_score_save_path(save_name)
    corpus_tf: Counter = when_raw_tf()
    avg_summarize_local_aligns(corpus_tf, input_file_path, save_path)


def when_tf_l1():
    input_file_name = "when_tf_l1.jsonl"
    input_file_path = get_attn_score_jsonl_path(input_file_name)
    save_name = "when_tf_l1.avg_score_summary"
    save_path = get_avg_score_save_path(save_name)
    corpus_tf: Counter = when_raw_tf()
    avg_summarize_local_aligns(corpus_tf, input_file_path, save_path)


if __name__ == "__main__":
    when_tf()
