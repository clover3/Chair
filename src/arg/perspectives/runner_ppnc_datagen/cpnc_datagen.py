from arg.perspectives.ppnc import cpnc_datagen
from arg.perspectives.ppnc import ppnc_datagen_50_perspective
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from arg.perspectives.ppnc.ppnc_worker import start_generate_jobs_for_train_val

# CPNR : Claim-Passage Neural Classifier

if __name__ == "__main__":
    name_prefix = "cpnc"

    def functor(cid_to_passage) -> CPPNCGeneratorInterface:
        return cpnc_datagen.Generator(cid_to_passage)

    start_generate_jobs_for_train_val(functor,
                                      ppnc_datagen_50_perspective.write_records,
                                      name_prefix)