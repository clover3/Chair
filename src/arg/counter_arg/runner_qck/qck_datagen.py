
from typing import List

from arg.counter_arg.eval import EvalCondition
# use top-k candidate as payload
from arg.counter_arg.header import splits
from arg.counter_arg.qck_datagen import start_job, load_base_resource
from arg.qck.decl import QKUnit
from cache import load_from_pickle, save_to_pickle


def load_qk(split) -> List[QKUnit]:
    return load_from_pickle("ca_qk_candidate_{}".format(split))


def save_to_cache():
    for split in splits:
        job_name = "argu_qck_datagen_{}".format(split)
        candidate_dict, correct_d = load_base_resource(EvalCondition.EntirePortalCounters, split)
        obj = candidate_dict, correct_d

        save_to_pickle(obj, job_name + "_base_resource")


def main():
    for split in splits:
        job_name = "argu_qck_datagen_{}".format(split)
        qk_candidate: List[QKUnit] = load_qk(split)
        candidate_dict, correct_d = load_from_pickle(job_name + "_base_resource")
        start_job(job_name,
                  split,
                  candidate_dict,
                  correct_d,
                  qk_candidate)



if __name__ == "__main__":
    main()
