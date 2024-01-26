import os
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
import sys
from omegaconf import OmegaConf

from misc_lib import TimeEstimatorOpt
from table_lib import tsv_iter
from taskman_client.wrapper3 import JobContext
from trainer_v2.per_project.transparency.mmp.pep.inf_helper import get_term_pair_predictor_fixed_context


def predict_with_fixed_context_model_and_save(
        model_path,
        payload_path,
        log_path,
):
    candidate_itr: List[List[str]] = list(tsv_iter(payload_path))
    predict_term_pairs = get_term_pair_predictor_fixed_context(model_path)
    out_f = open(log_path, "w")

    ticker = TimeEstimatorOpt(len(candidate_itr))
    for row in candidate_itr:
        q_term = row[0]
        d_terms = row[1:]
        payload = [(q_term, dt) for dt in d_terms]
        scores = predict_term_pairs(payload)
        assert len(scores) == len(d_terms)

        out_row = [q_term]
        for dt, score in zip(d_terms, scores):
            out_row.append(dt)
            out_row.append(score)

        out_f.write("\t".join(map(str, out_row)) + "\n")
        out_f.flush()
        ticker.tick()


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    job_no = int(sys.argv[2])

    with JobContext(f"Teacher_{job_no}"):
        predict_with_fixed_context_model_and_save(
            conf.model_path,
            os.path.join(conf.payload_dir, str(job_no) + ".txt"),
            os.path.join(conf.scored_terms_dir, str(job_no) + ".tsv"),
        )
    return NotImplemented


if __name__ == "__main__":
    main()