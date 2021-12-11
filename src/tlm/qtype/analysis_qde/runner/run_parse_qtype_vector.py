from cache import save_to_pickle
from cpath import at_output_dir
from tlm.qtype.analysis_qde.analysis_common import group_by_dim_threshold
from tlm.qtype.analysis_qde.parse_qtype_vector_pairwise import parse_q_weight_output


def main():
    run_name_list = ["mmd_4U"]
    for run_name in run_name_list:
        raw_prediction_path = at_output_dir("qtype", run_name)
        out_entries = parse_q_weight_output(raw_prediction_path)
        save_to_pickle(out_entries, run_name + "_qtype_parsed")
        out_entries_flat = []
        for e1, e2 in out_entries:
            out_entries_flat.append(e1)
            out_entries_flat.append(e2)
        group_by_dim_threshold(out_entries_flat)


if __name__ == "__main__":
    main()