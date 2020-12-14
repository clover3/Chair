from typing import Dict, NewType

ModelName = NewType('ModelName', str)
DataName = NewType('DataName', str)


def get_all_data_size() -> Dict[DataName, Dict]:
    data_size_dict: Dict[DataName, Dict] = dict()

    all_passage_pairwise_ex = DataName("all_passage_pointwise_ex")
    robust_all_passage = DataName("robust_all_passage")
    robust_all_passage_unpaired = DataName("robust_all_passage_unpaired")
    robust_selected = DataName("robust_selected")


    robust_K = ModelName("Robust_K")
    robust_L = ModelName("robust_L")

    model_name_to_data_name: Dict[ModelName, DataName] = {
        robust_K: all_passage_pairwise_ex,
        robust_L: robust_all_passage_unpaired
    }

    data_size_dict[all_passage_pairwise_ex] = {
        "651": 222338,
        "301": 467338,
        "601": 358720,
        "401": 821393,
        "351": 610591
    }
    data_size_dict[robust_all_passage_unpaired] = {
        "401": 1483830,
        "351": 1050250,
        "601": 597750,
        "301": 757794,
        "651": 279430,
    }
    data_size_dict[robust_all_passage] = {
        "351": 525125,
        "401": 741915,
        "601": 298875,
        "301": 378897,
        "651": 139715,
    }
    data_size_dict[robust_selected] = {
        "601": 355630,
        "351": 598290,
        "401": 809570,
        "651": 218648,
        "301": 457621,
    }
    return data_size_dict


def main():
    all_data_size_d = get_all_data_size()
    data_name = DataName("robust_selected")

    target_data = ["301", "351", "401", "601"]
    data_size_d = all_data_size_d[data_name]
    batch_size = 32
    selected_data_size = 0
    for name in target_data:
        selected_data_size += data_size_d[name]

    step_per_epoch = selected_data_size / batch_size
    print("Data size", selected_data_size)
    print("batch size", batch_size)
    print('step_per_epoch', step_per_epoch)


if __name__ == "__main__":
    main()