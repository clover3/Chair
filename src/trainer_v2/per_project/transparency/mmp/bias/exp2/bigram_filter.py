import sys

from omegaconf import OmegaConf

from table_lib import tsv_iter


def main():
    conf_path = sys.argv[1]
    config = OmegaConf.load(conf_path)
    generic_keywords = [line.lower().strip() for line in open(config.known_generic_terms_path, "r")]

    prev_annots = list(tsv_iter(config.annot_save_path))
    annotated_bigrams = []
    for bigram, in prev_annots:
        brand, word = bigram.split()
        annotated_bigrams.append(bigram)
        # if annot == "generic":
        #     generic_keywords.append(word)

    full_bigrams = [line.lower().strip() for line in open(config.src_bigram_path, "r")]
    # f = open(config.annot_save_path, "a")

    model_bigrams = []
    generic_keyword_set = set(generic_keywords)

    print(generic_keyword_set)
    return
    skip_idx = set()
    for j in range(0, len(full_bigrams)):
        if j not in skip_idx:
            try:
                brand, word = full_bigrams[j].split()
                if word in generic_keyword_set:
                    skip_idx.add(j)
            except ValueError:
                print(full_bigrams[j])
                raise

    print(generic_keyword_set)
    return

    for idx, s in enumerate(full_bigrams):
        if s in annotated_bigrams:
            continue
        if idx in skip_idx:
            continue
        print("Does this contain a model name?", s)
        r = input()
        if r == "y":
            model_bigrams.append(s)
            f.write(s + "\n")
        else:
            brand, generic = s.split()
            generic_keyword_set.add(generic)
            n_remain = 0
            for j in range(idx+1, len(full_bigrams)):
                if j not in skip_idx:
                    brand, word = full_bigrams[j].split()
                    if word in generic_keyword_set:
                        skip_idx.add(j)
                    else:
                        n_remain += 1
            print("{} remains".format(n_remain))


if __name__ == "__main__":
    main()