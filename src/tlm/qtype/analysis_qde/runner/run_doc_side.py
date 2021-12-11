from cache import load_from_pickle
from tlm.qtype.analysis_qde.analysis_common import caculate_cut


def main():
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype") # qtype_2T + Train all
    # doc_side_analysis(qtype_entries)
    caculate_cut(qtype_entries)

if __name__ == "__main__":
    main()
