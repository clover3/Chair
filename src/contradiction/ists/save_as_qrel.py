from contradiction.ists.save_path_helper import get_qrel_path
from dataset_specific.ists.parse import AlignmentPredictionList
from dataset_specific.ists.path_helper import load_ists_label
from dataset_specific.ists.split_info import ists_genre_list, ists_split_list
from trec.trec_parse import write_trec_relevance_judgement
from trec.types import TrecRelevanceJudgementEntry


def do_for_genre_split(genre, split):
    save_path = get_qrel_path(genre, split)
    labels: AlignmentPredictionList = load_ists_label(genre, split)
    all_judgments = []
    for problem_id, per_problem in labels:
        no_align_list1 = []
        no_align_list2 = []
        for alignment in per_problem:
            type_labels = [l.strip() for l in alignment.align_types]
            if "noali" not in type_labels:
                continue

            for token_id in alignment.chunk_token_id1:
                if token_id != 0:
                    no_align_list1.append(token_id)

            for token_id in alignment.chunk_token_id2:
                if token_id != 0:
                    no_align_list2.append(token_id)

        no_align_list1.sort()
        no_align_list2.sort()
        qid1 = f"noali_{problem_id}_1"
        qid2 = f"noali_{problem_id}_2"
        for token_id in no_align_list1:
            e = TrecRelevanceJudgementEntry(qid1, str(token_id), 1)
            all_judgments.append(e)

        for token_id in no_align_list2:
            e = TrecRelevanceJudgementEntry(qid2, str(token_id), 1)
            all_judgments.append(e)
    write_trec_relevance_judgement(all_judgments, save_path)


def main():
    for genre in ists_genre_list:
        for split in ists_split_list:
            if genre == "answers-students" and split == "train":
                continue
            try:
                do_for_genre_split(genre, split)
            except Exception:
                print(genre, split)
                raise


if __name__ == "__main__":
    main()