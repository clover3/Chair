import json
import os

from contradiction.medical_claims.token_tagging.path_helper import get_sbl_label_json_path
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem
from cpath import output_path
from list_lib import index_by_fn


def main():
    labels = json.load(open(get_sbl_label_json_path(), "r"))
    def get_problem_id(d):
        return d['group_no'], d['inner_idx']
    labels.sort(key=get_problem_id)
    problems = load_alamri_problem()
    problem_d = index_by_fn(lambda p: (p.group_no, p.inner_idx), problems)
    save_path = os.path.join(output_path, "alamri_annotation1", "label", "sel_by_longest.txt")
    f = open(save_path, "w")

    def write_line(s):
        f.write(s + "\n")

    for l in labels:
        print(get_problem_id(l))
        p = problem_d[get_problem_id(l)]
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()

        write_line("text1: " + p.text1)
        write_line("text2: " + p.text2)

        def get_tokens_str(tokens, indices):
            valid_indices = [i for i in indices if i < len(tokens)]
            invalid_indices = [i for i in indices if i >= len(tokens)]
            if invalid_indices:
                print(len(tokens), invalid_indices)
            return ", ".join([tokens[i] for i in valid_indices])
        for key in l['label']:
            tokens = tokens1 if key.startswith("prem") else tokens2
            write_line("{}: {}".format(key,  get_tokens_str(tokens, l['label'][key])))
        write_line("")


if __name__ == "__main__":
    main()