import os
from typing import List, Any

from arg.counter_arg_retrieval.build_dataset.run1.annotation.ca_rate_compare import common_dir
from list_lib import lmap, lflatten
from misc_lib import get_first, get_second
from misc_lib import group_by
from mturk.parse_util import ColumnName, Textbox, HITScheme, Checkbox
from mturk.parse_util import parse_file, HitResult
from stats.agreement import cohens_kappa


def get_ca6_scheme():
    inputs = [ColumnName("topic"), ColumnName("claims_base64"), ColumnName("doc_id")]

    answer_units = []
    for i in range(11):
        for c in ["S", "O"]:
            q = Checkbox("Q{}{}.on".format(i, c))
            answer_units.append(q)
        q_text = Textbox("Q{}R".format(i))
        answer_units.append(q_text)
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme


def main():
    hit_scheme = get_ca6_scheme()

    ca6_input = os.path.join(common_dir, "CA6", "Batch_4512731_batch_results.csv")
    hit_results = parse_file(ca6_input, hit_scheme)
    input_columns = list(hit_results[0].inputs.keys())

    filter_key = "O.on"
    answer_columns_binary = []
    for unit in hit_scheme.answer_units:
        if isinstance(unit, Checkbox) and filter_key in unit.name:
            answer_columns_binary.append(unit.name)

    def get_input_key(hit_result: HitResult):
        return tuple([hit_result.inputs[key] for key in input_columns])

    list_answers = []
    for key, entries in group_by(hit_results, get_input_key).items():
        for ans_col in answer_columns_binary:
            answers = [e.outputs[ans_col] for e in entries]
            list_answers.append(list(answers))
    print(list_answers)
    annot1: List[Any] = lmap(get_first, list_answers)
    annot2: List[Any] = lmap(get_second, list_answers)

    flat_answers = lflatten(list_answers)
    print("pos rate: {}".format(sum(flat_answers) / len(flat_answers)))
    print('1 vs 2', cohens_kappa(annot1, annot2))



if __name__ == "__main__":
    main()