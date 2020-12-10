import json
import sys

from arg.qck.multi_file_save_to_trec_form import multi_file_save_to_trec_form_by_range

if __name__ == "__main__":
    run_config = json.load(open(sys.argv[1], "r"))
    multi_file_save_to_trec_form_by_range(run_config["prediction_dir"],
         run_config["idx_range"],
         run_config
    )