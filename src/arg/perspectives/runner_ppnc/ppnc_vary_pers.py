from arg.perspectives.eval_helper import get_eval_candidate_as_pids
from arg.perspectives.ppnc import ppnc_datagen_50_perspective
from arg.perspectives.ppnc.ppnc_decl import PPNCGeneratorInterface
from arg.perspectives.ppnc.ppnc_worker import start_generate_jobs_for_train_val

if __name__ == "__main__":
    name_prefix = "ppnc_50_pers"

    def functor(cid_to_passage) -> PPNCGeneratorInterface:
        candidate_pers = dict(get_eval_candidate_as_pids("train"))
        return ppnc_datagen_50_perspective.Generator(cid_to_passage, candidate_pers)

    start_generate_jobs_for_train_val(functor,
                                      ppnc_datagen_50_perspective.write_records,
                                      name_prefix)