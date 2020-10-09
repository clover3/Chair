import pickle
from typing import List, Dict, Tuple

from arg.perspectives.doc_value_viewer.calculate_doc_score import calculate_score
from arg.qck.decl import qck_convert_map
from arg.qck.doc_value_calculator import DocValueParts
from arg.qck.dynamic_kdp.score_summarizer import load_baseline_score_d
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.util import load_run_config


def main():
    run_config: Dict = load_run_config()
    baseline_d: Dict[Tuple[str, str], float] = \
        load_baseline_score_d(run_config['baseline_info_path'],
                              run_config['baseline_score_path'],
                              run_config['is_baseline_from_pickle'])

    info_save_path = run_config['info_path']
    info = load_combine_info_jsons(info_save_path, qck_convert_map)
    score_save_path = run_config['score_path']
    # calculate score for each kdp
    str_data_id = True
    doc_score_parts: List[DocValueParts] = calculate_score(info, score_save_path, baseline_d, str_data_id)
    summary_save_path = run_config['save_path']
    pickle.dump(doc_score_parts, open(summary_save_path, "wb"))


if __name__ == "__main__":
    main()
