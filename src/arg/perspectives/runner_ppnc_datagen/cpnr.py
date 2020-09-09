from arg.perspectives.ppnc import cpnr_datagen
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from arg.perspectives.ppnc.ppnc_worker import start_generate_jobs_for_train_val

# CPNR : Claim-Passage Neural Ranker

if __name__ == "__main__":
    name_prefix = "cpnr"

    def functor(cid_to_passage) -> CPPNCGeneratorInterface:
        return cpnr_datagen.Generator(cid_to_passage)

    start_generate_jobs_for_train_val(functor,
                                      cpnr_datagen.write_records,
                                      name_prefix)