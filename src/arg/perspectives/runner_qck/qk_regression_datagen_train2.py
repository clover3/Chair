from typing import Dict, Tuple

from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs_for_val
# Make payload without any annotation
from arg.perspectives.runner_ppnc.save_doc_value import load_cppnc_doc_value_val
from arg.qck.decl import KDP, QCKQuery
from arg.qck.qk_regression_datagen import QKRegressionInstanceGenerator


def main():
    score_dictionary: Dict[Tuple[str, str, int], float] = load_cppnc_doc_value_val()
    max_score = 0.01
    min_score = -0.01

    def get_score(query: QCKQuery, passage: KDP):
        key = query.query_id, passage.doc_id, passage.passage_idx
        if key in score_dictionary:
            raw_value = score_dictionary[key]
            raw_value = min(max_score, raw_value)
            value = max(min_score, raw_value)
        else:
            print("Not found", key)
            value = min_score

        return value

    start_generate_jobs_for_val(QKRegressionInstanceGenerator(get_score), "qk_regression")


if __name__ == "__main__":
    main()