from arg.counter_arg.header import splits
from arg.counter_arg.same_debate import load_base_resource
from cache import save_to_pickle


def save_to_cache():
    for split in splits:
        job_name = "argu_debate_qck_datagen_{}".format(split)
        candidate_dict, correct_d = load_base_resource(split)
        obj = candidate_dict, correct_d
        save_to_pickle(obj, job_name + "_base_resource")