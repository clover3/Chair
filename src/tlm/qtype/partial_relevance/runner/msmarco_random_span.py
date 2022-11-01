import random

from cache import load_pickle_from
from cpath import at_output_dir


def main():
    save_path = at_output_dir("qtype", "msmarco_random_spans.pickle")
    data = load_pickle_from(save_path)

    all_spans = []
    for per_doc in data:
        for item in per_doc:
            tokens, _, _ = item
            span = " ".join(tokens)
            all_spans.append(span)

    random.shuffle(all_spans)
    first100 = all_spans[:100]

    txt_save_path = at_output_dir("qtype", "msmarco_random_spans.txt")
    f = open(txt_save_path, "w")
    for item in first100:
        f.write("{}\n".format(item))


if __name__ == "__main__":
    main()