from itertools import chain
from typing import Dict, List, Tuple, NamedTuple, Iterator

from data_generator.argmining.ukp_header import all_topics
from data_generator.data_parser import ukp as ukp
from list_lib import l_to_map, lfilter, dict_value_map, lmap, flatten


class UkpDataPoint(NamedTuple):
    id: int
    topic: str
    sentence: str
    label: str
    split: str

    @classmethod
    def from_dict(cls, d: Dict):
        return UkpDataPoint(
            id=d['dp_id'],
            topic=d['topic'],
            sentence=d['sentence'],
            label=d['annotation'],
            split=d['set']
        )


def load_all_data() -> Tuple[Dict[str, List[UkpDataPoint]], Dict[str, List[UkpDataPoint]]]:
    all_data: Dict[str, List[Dict]] = l_to_map(ukp.load, all_topics)

    # split train / dev
    def is_train(entry: Dict) -> bool:
        return entry['set'] == 'train'

    def is_validation(entry: Dict) -> bool:
        return entry['set'] == 'val'

    def filter_train(data: List[Dict]) -> List[Dict]:
        return lfilter(is_train, data)

    def filter_validation(data: List[Dict]) -> List[Dict]:
        return lfilter(is_validation, data)

    raw_train_data: Dict[str, List[Dict]] = dict_value_map(filter_train, all_data)
    raw_val_data: Dict[str, List[Dict]] = dict_value_map(filter_validation, all_data)

    def all_data_iterator() -> Iterator[Dict]:
        for data_list in chain(raw_train_data.values(), raw_val_data.values()):
            for dp in data_list:
                yield dp

    dp_id = 1
    for dp in all_data_iterator():
        dp['dp_id'] = dp_id
        dp_id += 1

    def to_data_point(l : List[Dict]) -> List[UkpDataPoint]:
        return lmap(UkpDataPoint.from_dict, l)

    train_data = dict_value_map(to_data_point, raw_train_data)
    val_data = dict_value_map(to_data_point, raw_val_data)

    return train_data, val_data


def load_all_data_flat() -> List[UkpDataPoint]:
    train_data, val_data = load_all_data()
    l = []
    l.extend(flatten(train_data.values()))
    l.extend(flatten(val_data.values()))
    return l


class SplitLoader:
    def __init__(self):
        pass



def load_dataset_by_split(split: str) -> Dict[str, List[Dict]]:
    all_data: Dict[str, List[Dict]] = l_to_map(ukp.load, all_topics)

    # split train / dev
    def is_train(entry: Dict) -> bool:
        return entry['set'] == 'train'

    def is_validation(entry: Dict) -> bool:
        return entry['set'] == 'val'

    def filter_train(data: List[Dict]) -> List[Dict]:
        return lfilter(is_train, data)

    def filter_validation(data: List[Dict]) -> List[Dict]:
        return lfilter(is_validation, data)

    if split == "train":
        train_data: Dict[str, List[Dict]] = dict_value_map(filter_train, all_data)
        return train_data
    elif split == "dev":
        val_data: Dict[str, List[Dict]] = dict_value_map(filter_validation, all_data)
        return val_data
    else:
        assert False