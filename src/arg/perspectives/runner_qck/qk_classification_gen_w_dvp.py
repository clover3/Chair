import pickle

from arg.perspectives.runner_qck.qcknc_common import do_all_jobs
from arg.qck.decl import QCKQuery, KDP
from arg.qck.dvp_to_correctness import dvp_to_correctness
from arg.qck.qknc_datagen import QKInstanceGenerator
# Make payload without any annotation
from arg.util import load_run_config


def main():
    config = load_run_config()
    dvp_list = pickle.load(open(config['dvp_path'], "rb"))
    job_prefix = config['job_prefix']
    qk_candidate_name = config['qk_candidate_name']

    config = {}
    dvp_to_correctness_dict = dvp_to_correctness(dvp_list, config)

    def is_correct(q: QCKQuery, kdp: KDP) -> int:
        key = q.query_id, (kdp.doc_id, kdp.passage_idx)
        return bool(dvp_to_correctness_dict[key])

    do_all_jobs(QKInstanceGenerator(is_correct),
                                qk_candidate_name,
                                job_prefix, "val")


if __name__ == "__main__":
    main()
