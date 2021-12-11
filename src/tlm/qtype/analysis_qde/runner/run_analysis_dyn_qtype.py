from cache import load_from_pickle
from tlm.qtype.analysis_fixed_qtype.parse_dyn_qtype_vector import show_qtype_embeddings


def main():
    # pred_path = sys.argv[1]
    # info_path = sys.argv[2]
    split = "train"
    # qtype_entries, query_info_dict = load_parse(info_path, pred_path, split)
    # obj = qtype_entries, query_info_dict
    # save_to_pickle(obj, "run_analysis_dyn_qtype")
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    known_qtype_ids = show_qtype_embeddings(qtype_entries, query_info_dict, split)
    # run_qtype_analysis(qtype_entries, query_info_dict, known_qtype_ids)


if __name__ == "__main__":
    main()
