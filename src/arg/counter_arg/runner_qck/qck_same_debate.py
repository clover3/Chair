from typing import List

# use top-k candidate as payload
from arg.counter_arg.header import splits
from arg.counter_arg.qck_datagen import start_job
from arg.qck.decl import QKUnit
from cache import load_from_pickle
from misc_lib import tprint


def load_qk(split) -> List[QKUnit]:
    return load_from_pickle("ca_qk_candidate_{}".format(split))


def main():
    print("Process started")
    for split in splits:
        tprint("Loading pickles")
        job_name = "argu_debate_qck_datagen_{}".format(split)
        qk_candidate: List[QKUnit] = load_qk(split)
        candidate_dict, correct_d = load_from_pickle(job_name + "_base_resource")
        tprint("Starting job")
        start_job(job_name,
                  split,
                  candidate_dict,
                  correct_d,
                  qk_candidate)


if __name__ == "__main__":
    #save_to_cache()
    main()
