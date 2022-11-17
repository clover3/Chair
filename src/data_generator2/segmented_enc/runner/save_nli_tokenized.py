import pickle

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.mnli.mnli_reader import MNLIReader
from data_generator.NLI.nli_info import nli_tokenized_path


def do_for_split(split):
    reader = MNLIReader()
    tokenizer = get_tokenizer()
    def convert(text):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    items = []
    for e in reader.load_split(split):
        tokens1 = convert(e.premise)
        tokens2 = convert(e.hypothesis)
        label = e.get_label_as_int()
        items.append((tokens1, tokens2, label))

    save_path = nli_tokenized_path(split)
    pickle.dump(items, open(save_path, "wb"))
    return items


def main():
    do_for_split("dev")
    do_for_split("train")


if __name__ == "__main__":
    main()
