import json
import os

from contradiction.medical_claims.biobert.voca_common import get_biobert_tokenizer
from contradiction.medical_claims.token_tagging.solvers.deletion_ranker import make_ranked_list_inner
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from trec.trec_parse import write_trec_ranked_list_entry


def make_ranked_list(dir_path, save_name, info_path, tag_type, tokenizer):
    save_path = os.path.join(output_path, "alamri_annotation1", "ranked_list", save_name + ".txt")
    info_d = json.load(open(info_path, "r", encoding="utf-8"))
    all_ranked_list = make_ranked_list_inner(dir_path, info_d, tag_type, tokenizer)
    write_trec_ranked_list_entry(all_ranked_list, save_path)



def main_for_bert():
    # tag_type = "mismatch"
    tag_type = "conflict"
    dir_path = os.path.join(output_path, "bert_alamri1_deletion_8")
    save_name = "deletion_" + tag_type
    info_path = os.path.join(output_path, "alamri_annotation1", "tfrecord", "bert_alamri1.info")
    tokenizer = get_tokenizer()
    make_ranked_list(dir_path, save_name, info_path, tag_type, tokenizer)


def main_for_biobert():
    tag_type = "mismatch"
    # tag_type = "conflict"
    tokenizer = get_biobert_tokenizer()

    dir_path = os.path.join(output_path, "biobert_alamri1_deletion")
    save_name = "biobert_deletion_" + tag_type
    info_path = os.path.join(output_path, "alamri_annotation1", "tfrecord", "biobert_alamri1.info")
    make_ranked_list(dir_path, save_name, info_path, tag_type, tokenizer)


if __name__ == "__main__":
    main_for_bert()