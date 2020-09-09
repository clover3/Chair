from arg.perspectives.ppnc import ppnc_datagen
from arg.perspectives.ppnc.ppnc_worker import start_generate_jobs_for_train_val


if __name__ == "__main__":
    name_prefix = "passage_pers_classifier"
    start_generate_jobs_for_train_val(ppnc_datagen.Generator, ppnc_datagen.write_records, name_prefix)