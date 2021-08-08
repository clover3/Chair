from arg.counter_arg_retrieval.build_dataset.run1.enum_disagreeing_ones import load_ca3_master
from arg.counter_arg_retrieval.build_dataset.verify_common import summarize_agreement
from misc_lib import group_by
from mturk.parse_util import HitResult


def main():
    hit_results = load_ca3_master()
    input_columns = list(hit_results[0].inputs.keys())
    answer_columns = list(hit_results[0].outputs.keys())
    def get_input_key(hit_result: HitResult):
        return tuple([hit_result.inputs[key] for key in input_columns])

    answer_list_d = summarize_agreement(hit_results, 0)
    n_answers = len(answer_list_d["Q1."])
    expected_len = 3
    for key, entries in group_by(hit_results, get_input_key).items():
        row = []
        row.extend(key)
        for ans_col in answer_columns:
            l = [e.outputs[ans_col] for e in entries]
            if len(l) < expected_len:
                l.extend(["-"] * (expected_len - len(l)))
            row += l

        out_row = "\t".join(map(str, row))
        print(out_row)



if __name__ == "__main__":
    main()