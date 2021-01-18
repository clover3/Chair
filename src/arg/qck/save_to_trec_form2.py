import json
import sys

from arg.qck.save_to_trec_form import save_to_common_path

if __name__ == "__main__":
    run_config = json.load(open(sys.argv[1], "r"))
    save_to_common_path(run_config["prediction_path"],
                        run_config["info_path"],
                        run_config["run_name"],
                        run_config["input_type"],
                        run_config["max_entry"],
                        run_config["combine_strategy"],
                        run_config["score_type"],
                        True
    )