from collections import Counter

import spacy
from spacy.tokens import Doc

from dataset_specific.mnli.snli_reader import SNLIReader
from trainer_v2.custom_loop.per_task.slr.segmentation import spacy_segment


# For hypo len <=4 covers 96%,
# len <= 8 covers 99%
# For H, Avg span = 2.32


def main():
    reader = SNLIReader()
    nlp = spacy.load("en_core_web_sm")
    p_counter = Counter()
    h_counter = Counter()
    n = 0
    n_target = 1000
    for e in reader.load_split("train"):
        todo = [("premise", e.premise, p_counter), ("hypothesis", e.hypothesis, h_counter)]
        for sent_type, sent, counter in todo:
            doc: Doc = nlp(sent)
            l = len(spacy_segment(doc))
            counter[l] += 1
        n += 1
        if n > n_target:
            break

    todo = [("premise", p_counter), ("hypothesis", h_counter)]

    for sent_type, counter in todo:
        print(sent_type)
        total = sum(counter.values())
        s = 0
        s_d = 0
        accum = 0
        for i in range(20):
            s += counter[i] * i
            s_d += counter[i]
            accum += counter[i]
            print("{0}: {1:.3f}".format(i, accum / total))

        print("Avg # {0:.2f}".format(s / s_d))


if __name__ == "__main__":
    main()