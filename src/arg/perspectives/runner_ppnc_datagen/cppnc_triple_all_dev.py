from arg.perspectives.eval_caches import get_eval_candidate_as_pids
from arg.perspectives.ppnc import cppnc_triple_datagen, cppnc_datagen
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from arg.perspectives.ppnc.ppnc_worker import start_generate_jobs_for_dev

# cppnc : Claim-Perspective-Passage Neural Classifier

if __name__ == "__main__":
    name_prefix = "cppnc_triple_all"

    def functor(cid_to_passage) -> CPPNCGeneratorInterface:
        candidate_pers = dict(get_eval_candidate_as_pids("dev"))
        return cppnc_datagen.Generator(cid_to_passage, candidate_pers, False)

    start_generate_jobs_for_dev(functor,
                              cppnc_triple_datagen.write_records,
                              name_prefix)