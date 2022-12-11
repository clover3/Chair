from collections import Counter

from contradiction.medical_claims.cont_classification.path_helper import load_cont_classification_problems, \
    load_problem_notes


def main():
    for split in ["dev", "test"]:
        print(split)
        counter = Counter()
        problems = load_cont_classification_problems(split)
        note_dict = load_problem_notes(split)
        for p in problems:
            counter[p.label] += 1
            note_text = note_dict[p.signature()]
            counter[note_text] += 1

        print(counter)


if __name__ == "__main__":
    main()
