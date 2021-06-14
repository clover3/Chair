import json
import random

from arg.counter_arg_retrieval.build_dataset.resources import load_step1_claims
from cpath import at_output_dir


def main():
    topics = load_step1_claims()

    topics2 = random.choices(topics, k=20)

    save_path = at_output_dir("ca_building", "claims.step2.txt")
    json.dump(topics2, open(save_path, "w"), indent=True)


if __name__ == "__main__":
    main()
