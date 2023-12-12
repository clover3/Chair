import sys

from omegaconf import OmegaConf

from table_lib import tsv_iter


def main():
    conf_path = sys.argv[1]
    config = OmegaConf.load(conf_path)
    generic_keywords = [line.lower().strip() for line in open(config.known_generic_terms_path, "r")]

    prev_annots = list(tsv_iter(config.annot_save_path))
    model_name_bigrams = []
    for bigram, in prev_annots:
        model_name_bigrams.append(bigram)

    full_bigrams = [line.lower().strip() for line in open(config.src_bigram_path, "r")]
    f = open(config.annot_save_path, "a")
    generic_keywords = set(generic_keywords)
    for j in range(0, len(full_bigrams)):
        if full_bigrams[j] in model_name_bigrams:
            f.write(f"{full_bigrams[j]}\tbrand\n")
        else:
            f.write(f"{full_bigrams[j]}\tgeneric\n")
            brand, word = full_bigrams[j].split()
            generic_keywords.add(word)


        if full_bigrams[j] == "cadillac escalade":
            break

    print(generic_keywords)


if __name__ == "__main__":
    main()