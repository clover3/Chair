import csv
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from misc_lib import group_by
from mturk.parse_util import HitResult
from tab_print import print_table


def consistency_check(hits: List[HitResult]):
    # returns key, value
    # The instance with same key should have same value
    keys = "c_text", "doc_id", "passage_idx", "passage"
    values_columns = (
        "Mention_C",
        "Arg_Support_C",
        "Arg_Oppose_C",
        "Info_Support_C",
        "Info_Oppose_C"
    )
    same_claim_passage_constraint = keys, values_columns

    keys = "c_text", "p_text"
    values_columns = (
        "P_Support_C",
        "P_Oppose_C",
    )
    same_claim_perspective_constraint = keys, values_columns

    todo = [same_claim_perspective_constraint,
            same_claim_passage_constraint]

    out_payload = []
    for keys, values_columns in todo:
        def get_keys(h: HitResult):
            return tuple(h.inputs[k] for k in keys)
        grouped = group_by(hits, get_keys)
        for value in values_columns:
            print("With {} -> {} constraints".format(keys, value))
            for group_key, items in grouped.items():
                counter = Counter()
                for i in items:
                    counter[i.outputs[value]] += 1

                if len(counter) == 1:
                    pass
                else:
                    print(group_key)
                    table = []
                    relevant_columns = ['assignment_id'] + list(keys) + [value]
                    table.append(relevant_columns)
                    for i in items:
                        inputs = [i.inputs[k] for k in keys]
                        outputs = [i.outputs[value]]
                        row = [i.assignment_id] + inputs + outputs
                        table.append(row)
                    print_table(table)
                    out_payload.extend(table)
                    out_payload.append([])

    return out_payload


def apply_patch(hits: List[HitResult], patch_data: Dict[str, List[Tuple[str, str]]]):
    n_modify = 0
    n_hit_w_modify = 0
    for h in hits:
        any_modify = False
        if h.assignment_id in patch_data:
            changes = patch_data[h.assignment_id]
            for out_column, new_value_raw in changes:
                original_value = h.outputs[out_column]
                if type(original_value) == int:
                    new_value = int(new_value_raw)
                else:
                    new_value = new_value_raw

                if original_value != new_value:
                    h.outputs[out_column] = new_value
                    n_modify += 1
                    any_modify = True
        if any_modify:
            n_hit_w_modify += 1
    print("{} of {} hits are modified. Total of {} fields are updated".format(
        n_hit_w_modify, len(hits), n_modify))


def load_patch_data(patch_data_path):
    reader = csv.reader(open(patch_data_path, "r"))
    OUT_TABLE = 1
    IN_TABLE = 2
    state = OUT_TABLE
    cur_table = []
    table_list = []
    def strip(row):
        first_empty = None
        for idx, v in enumerate(row):
            if v:
                assert first_empty is None
            else:
                if first_empty is None:
                    first_empty = idx
        return row[:first_empty]


    for row in reader:
        row = strip(row)
        if state == OUT_TABLE:
            if row:
                cur_table.append(row)
                state = IN_TABLE
            else:
                pass
        elif state == IN_TABLE:
            if row:
                cur_table.append(row)
            else:
                table_list.append(cur_table)
                cur_table = []
                state = OUT_TABLE

    if cur_table:
        table_list.append(cur_table)

    patches = defaultdict(list)
    for table in table_list:
        head = table[0]
        content = table[1:]
        assert head[0] == "assignment_id"
        assert len(head) == len(content[0])
        assert len(head) == len(content[-1])
        for row in content:
            assignment_id = row[0]
            out_column = head[-1]
            assert out_column
            new_value = row[-1]
            assert new_value
            patches[assignment_id].append((out_column, new_value))
    return patches