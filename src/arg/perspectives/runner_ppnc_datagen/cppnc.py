from arg.perspectives.eval_helper import get_eval_candidate_as_pids
from arg.perspectives.ppnc import cppnc_datagen
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from arg.perspectives.ppnc.ppnc_worker import start_generate_jobs_for_train_val

# cppnc : Claim-Perspective-Passage Neural Classifier

if __name__ == "__main__":
    name_prefix = "cppnc"

    def functor(cid_to_passage) -> CPPNCGeneratorInterface:
        candidate_pers = dict(get_eval_candidate_as_pids("train"))
        return cppnc_datagen.Generator(cid_to_passage, candidate_pers)

    start_generate_jobs_for_train_val(functor,
                                      cppnc_datagen.write_records,
                                      name_prefix)