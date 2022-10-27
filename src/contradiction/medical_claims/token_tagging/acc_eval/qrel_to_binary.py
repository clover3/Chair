from typing import List, Tuple

# TODO load trec_qrel
from cache import save_list_to_jsonl_w_fn
from contradiction.medical_claims.token_tagging.path_helper import get_sbl_qrel_path, get_sbl_binary_label_path
from contradiction.token_tagging.acc_eval.label_loaders import SentTokenLabel
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem
from trec.qrel_parse import load_qrels_flat_per_query
from trec.types import DocID


def convert_qrel_to_acc(judgment_path, tag):
    qrels = load_qrels_flat_per_query(judgment_path)
    problems = load_alamri_problem()
    parsed_labels: List[SentTokenLabel] = []
    for p in problems:
        sent_names = ["prem", "hypo"]
        text_list = [p.text1, p.text2]

        for i in [0, 1]:
            qid = "{}_{}_{}".format(p.get_problem_id(), sent_names[i], tag)
            try:
                entries: List[Tuple[DocID, int]] = qrels[qid]
                text = text_list[i]
                n_tokens = len(text.split())
                label_arr = [0 for _ in range(n_tokens)]
                for doc_id, label in entries:
                    try:
                        label_arr[int(doc_id)] = label
                    except IndexError:
                        print("text has {} tokens but got {}".format(n_tokens, doc_id))
                parsed_labels.append(SentTokenLabel(qid, label_arr))
            except KeyError:
                pass
    return parsed_labels


def main():
    for split in ["val", "test"]:
        for tag in ["conflict", "mismatch"]:
                qrel_path = get_sbl_qrel_path(split)
                parsed_labels = convert_qrel_to_acc(qrel_path, tag)
                save_path = get_sbl_binary_label_path(tag, split)
                save_list_to_jsonl_w_fn(parsed_labels, save_path, SentTokenLabel.to_json)


if __name__ == "__main__":
    main()