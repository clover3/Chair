import sys


def get_steps(topic, split, batch_size):
    info = {"dev_abortion": 315,
            "dev_cloning": 243,
            "dev_death_penalty": 293,
            "dev_gun_control": 268,
            "dev_marijuana_legalization": 198,
            "dev_minimum_wage": 198,
            "dev_nuclear_energy": 286,
            "dev_school_uniforms": 241,
            "train_abortion": 15514,
            "train_cloning": 16154,
            "train_death_penalty": 15714,
            "train_gun_control": 15937,
            "train_marijuana_legalization": 16561,
            "train_minimum_wage": 16563,
            "train_nuclear_energy": 15768,
            "train_school_uniforms": 16176,
            "tdev_abortion": 13963,
            "tdev_cloning": 14539,
            "tdev_death_penalty": 14143,
            "tdev_gun_control": 14344,
            "tdev_marijuana_legalization":  14905,
            "tdev_minimum_wage":  14907,
            "tdev_nuclear_energy":  14192,
            "tdev_school_uniforms":  14559,
            "ttrain_abortion":  13963,
            "ttrain_cloning":  14539,
            "ttrain_death_penalty":  14143,
            "ttrain_gun_control":  14344,
            "ttrain_marijuana_legalization":  14905,
            "ttrain_minimum_wage":  14907,
            "ttrain_nuclear_energy":  14192,
            "ttrain_school_uniforms":  14559,
            }

    key = "{}_{}".format(split, topic)
    num_data = info[key]
    steps = int(num_data / int(batch_size))
    print(steps, end="")

if __name__ == "__main__":
    get_steps(sys.argv[1], sys.argv[2], sys.argv[3])