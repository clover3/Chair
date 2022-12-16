import os
from typing import NamedTuple, Iterator
import tensorflow_datasets as tfds

from data_generator.NLI.enlidef import nli_label_list


class NLIPairData(NamedTuple):
    premise: str
    hypothesis: str
    label: str
    data_id: str

    def get_label_as_int(self):
        return nli_label_list.index(self.label)


class NLIByTFDS:
    def __init__(self, dataset_name, label_list):
        self.all_dataset = tfds.load(name=dataset_name)
        self.label_list = label_list

    def get_train(self) -> Iterator[NLIPairData]:
        return self.load_split("train")

    def get_dev(self) -> Iterator[NLIPairData]:
        return self.load_split("validation")

    def load_split(self, split_name) -> Iterator[NLIPairData]:
        def fetch_str(e, key):
            return e[key].numpy().decode()
        dataset = self.all_dataset[split_name]
        for idx, item in enumerate(dataset):
            new_d = {}
            for seg_name in ["premise", "hypothesis"]:
                new_d[seg_name] = item[seg_name].numpy().decode()
            label = int(item['label'].numpy())
            data_id = f"{split_name}_{idx}"
            yield NLIPairData(
                fetch_str(item, 'premise'),
                fetch_str(item, 'hypothesis'),
                self.label_list[label],
                data_id)

    def get_data_size(self, split_name):
        dataset = self.all_dataset[split_name]
        return len(dataset)


class SNLIReaderTFDS(NLIByTFDS):
    def __init__(self):
        dataset_name = "snli"
        super(SNLIReaderTFDS, self).__init__(dataset_name, nli_label_list)
