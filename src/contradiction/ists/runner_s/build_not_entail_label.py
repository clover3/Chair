from typing import List

from contradiction.ists.not_entail_convertor import convert_to_sent_token_label, sent_token_label_to_trec_qrel
from contradiction.ists.save_path_helper import get_not_entail_label_path, get_not_entail_qrel_path
from contradiction.token_tagging.acc_eval.parser import save_sent_token_label, load_sent_token_label
from dataset_specific.ists.parse import iSTSProblem
from dataset_specific.ists.path_helper import load_ists_problems, load_ists_label
from dataset_specific.ists.split_info import ists_enum_split_genre_combs
from trec.trec_parse import write_trec_relevance_judgement
from trec.types import TrecRelevanceJudgementEntry


def do_for_genre_split(genre, split):
    print(genre, split)
    problems: List[iSTSProblem] = load_ists_problems(genre, split)
    labels = load_ists_label(genre, split)
    sent_token_labels = convert_to_sent_token_label(problems, labels)
    save_path = get_not_entail_label_path(genre, split)
    print(save_path)
    save_sent_token_label(sent_token_labels, save_path)


def to_trec_entries(genre, split):
    src_path = get_not_entail_label_path(genre, split)
    entries: List[TrecRelevanceJudgementEntry] = sent_token_label_to_trec_qrel(load_sent_token_label(src_path))
    save_path = get_not_entail_qrel_path(genre, split)
    write_trec_relevance_judgement(entries, save_path)



def main():
    for split, genre in ists_enum_split_genre_combs():
        do_for_genre_split(genre, split)
        to_trec_entries(genre, split)


if __name__ == "__main__":
    main()