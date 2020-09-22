from arg.perspectives.eval_caches import get_eval_candidate_as_pids
from arg.perspectives.ppnc import multi_evidence
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from arg.perspectives.ppnc.ppnc_worker import start_generate_jobs_for_train_val, start_generate_jobs_for_dev

# cppnc : Claim-Perspective-Passage Neural Classifier

if __name__ == "__main__":
    name_prefix = "cppnc_multi_evidence"

    def functor(cid_to_passage) -> CPPNCGeneratorInterface:
        candidate_pers = dict(get_eval_candidate_as_pids("train"))
        return multi_evidence.Generator(cid_to_passage, candidate_pers, False)

    num_window = 8

    def write_records_fn(records, max_seq_length, output_path):
        d_max_seq_length = num_window * max_seq_length
        multi_evidence.write_records(records, max_seq_length, d_max_seq_length, output_path)

    start_generate_jobs_for_train_val(functor,
                                      write_records_fn,
                                      name_prefix)

    def functor(cid_to_passage) -> CPPNCGeneratorInterface:
        candidate_pers = dict(get_eval_candidate_as_pids("dev"))
        return multi_evidence.Generator(cid_to_passage, candidate_pers, False)

    start_generate_jobs_for_dev(functor, write_records_fn, name_prefix)