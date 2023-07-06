from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_split
from typing import List, Iterable, Callable, Dict, Tuple, Set
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from list_lib import index_by_fn, foreach
from misc_lib import path_join
from cpath import output_path
import csv


def iter_rows(columns, problems, split):
    label_d_per_tag: Dict[str, Dict[str, SentTokenLabel]] = {}
    for tag in ["mismatch", "conflict"]:
        labels: List[SentTokenLabel] = load_sbl_binary_label(tag, split)
        print(f"{len(labels)} loaded for {tag}")
        label_d = index_by_fn(lambda x: x.qid, labels)
        label_d_per_tag[tag] = label_d

    seen = set()
    for p in problems:
        assert p.data_id not in seen
        seen.add(p.data_id)
        n_tokens1 = len(p.text1.split())
        n_tokens2 = len(p.text2.split())
        out_d = {
            "pair_id": p.get_problem_id(),
            "group_no": p.group_no,
            "sent1": p.text1,
            "sent2": p.text2,
        }

        def get_label(sent_type, tag) -> List[int]:
            query_id = p.get_problem_id() + "_{}_{}".format(sent_type, tag)
            label = label_d_per_tag[tag][query_id]
            return label.labels

        any_label = False
        for sent_type_in in ["prem", "hypo"]:
            sent_type_out: str = {
                'prem': "sent1",
                'hypo': "sent2",
            }[sent_type_in]
            sent_len = {
                'prem': n_tokens1,
                'hypo': n_tokens2
            }[sent_type_in]
            for tag in ["mismatch", "conflict"]:
                try:
                    label: List[int] = get_label(sent_type_in, tag)
                    assert len(label) == sent_len
                    any_label = True
                except KeyError:
                    label = [0] * sent_len

                tag_out = {
                    "mismatch": "neutral",
                    "conflict": "contradiction"
                }[tag]
                out_column: str = f"{sent_type_out}_{tag_out}"
                out_d[out_column] = " ".join(map(str, label))

        if not any_label:
            continue

        out_val = []
        for i in range(len(columns)):
            key = columns[i]
            out_val.append(out_d[key])
        yield out_val


def main():
    columns = ["pair_id", "group_no",
               "sent1", "sent2",
               "sent1_contradiction", "sent1_neutral",
               "sent2_contradiction", "sent2_neutral",
               ]
    for split in ["val", "test"]:
        problems = load_alamri_split(split)
        save_path = path_join(output_path, "alamri_annotation1", f"{split}.csv")
        csv_writer = csv.writer(open(save_path, "w", newline='', encoding="utf-8"))
        csv_writer.writerow(columns)
        rows = list(iter_rows(columns, problems, split))
        print(f"{split} {len(rows)} rows")
        foreach(csv_writer.writerow, rows)


if __name__ == "__main__":
    main()