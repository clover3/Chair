import sys

from tlm.qtype.analysis_fixed_qtype.parse_dyn_qtype_vector import run_qtype_analysis


def main():
    pred_path = sys.argv[1]
    info_path = sys.argv[2]
    run_qtype_analysis(pred_path, info_path, "dev")



if __name__ == "__main__":
    main()