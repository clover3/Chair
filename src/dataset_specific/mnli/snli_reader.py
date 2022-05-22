from typing import Iterator

from data_generator.data_parser.esnli import load_split
from dataset_specific.mnli.mnli_reader import NLIPairData


def e_snli_parse(d) -> NLIPairData:
    return NLIPairData(
        premise=d["Sentence1"],
        hypothesis=d["Sentence2"],
        label=d["gold_label"],
        data_id=d['pairID']
    )


class SNLIReader:
    def get_train(self) -> Iterator[NLIPairData]:
        return self.load_split("train")

    def get_dev(self) -> Iterator[NLIPairData]:
        return self.load_split("dev")

    def load_split(self, split_name) -> Iterator[NLIPairData]:
        dict_list = load_split(split_name)
        nli_pair_data: Iterator[NLIPairData] = map(e_snli_parse, dict_list)
        return nli_pair_data


def main():
    reader = SNLIReader()
    for e in reader.get_train():
        print(e)
        break

    print(sum(1 for _ in reader.get_train()))


if __name__ == "__main__":
    main()