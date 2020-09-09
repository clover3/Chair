from arg.perspectives.eval_helper import get_eval_candidate_as_pids
from arg.perspectives.ppnc import pdcd_datagen
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from arg.perspectives.ppnc.ppnc_worker import start_generate_jobs_for_train_val

# PDCD: Perspective-Doc with Claim-Doc

if __name__ == "__main__":
    name_prefix = "pdcd"

    def functor(cid_to_passage) -> CPPNCGeneratorInterface:
        candidate_pers = dict(get_eval_candidate_as_pids("train"))
        return pdcd_datagen.Generator(cid_to_passage, candidate_pers)

    start_generate_jobs_for_train_val(functor,
                                      pdcd_datagen.write_records,
                                      name_prefix)