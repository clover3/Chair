from typing import Dict, NewType

ModelName = NewType('ModelName', str)
DataName = NewType('DataName', str)


def get_all_data_size() -> Dict[DataName, Dict]:
    data_size_dict: Dict[DataName, Dict] = dict()

    all_passage_pairwise_ex = DataName("all_passage_pointwise_ex")
    robust_all_passage = DataName("robust_all_passage")
    robust_all_passage_unpaired = DataName("robust_all_passage_unpaired")
    robust_selected = DataName("robust_selected")
    robust_all_passage_predict = DataName("robust_all_passage_predict")
    robust_train_desc = DataName("robust_train_desc")
    robust_neg100 = DataName("robust_neg100")
    robust_pairwise_head = DataName("robust_pairwise_head")
    robust_pairwise_pos_repeat = DataName("robust_pairwise_pos_repeat")
    first_512_desc = DataName("first_512_desc")


    robust_K = ModelName("Robust_K")
    robust_L = ModelName("robust_L")


    model_name_to_data_name: Dict[ModelName, DataName] = {
        robust_K: all_passage_pairwise_ex,
        robust_L: robust_all_passage_unpaired
    }

    data_size_dict[all_passage_pairwise_ex] = {
        "301": 467338,
        "351": 610591,
        "401": 821393,
        "601": 358720,
        "651": 222338,
    }
    data_size_dict[robust_all_passage_unpaired] = {
        "301": 757794,
        "351": 1050250,
        "401": 1483830,
        "601": 597750,
        "651": 279430,
    }
    data_size_dict[robust_all_passage] = {
        "301": 378897,
        "351": 525125,
        "401": 741915,
        "601": 298875,
        "651": 139715,
    }
    data_size_dict[robust_selected] = {
        "301": 457621,
        "351": 598290,
        "401": 809570,
        "601": 355630,
        "651": 218648,
    }
    data_size_dict[robust_all_passage_predict] = {
        "601": 14331,
        "401": 17934,
        "351": 16215,
        "651": 16007,
        "301": 14740,
    }
    data_size_dict[robust_neg100] = {
        "301": 95744,
        "351": 127378,
        "401": 154456,
        "601": 73605,
        "651": 46715
    }
    data_size_dict[robust_train_desc] = {
        "301": 133392,
        "351": 161762,
        "401": 186594,
        "601": 104097,
        "651": 82839,
    }
    data_size_dict[robust_pairwise_head] = {
        "301": 4290,
        "651": 2062,
        "401": 4728,
        "351": 4674,
        "601": 1658,
    }
    data_size_dict[robust_pairwise_pos_repeat] = {
        "351": 23370,
        "401": 23640,
        "601": 8290,
        "301": 21450,
        "651": 10310,
    }
    data_size_dict[first_512_desc] = {
        "651" : 66685,
        "301" : 94287,
        "601" : 74515,
        "351" : 109780,
        "401" : 114020,
    }
    data_size_dict["robust_pairwise_head_desc_neg_major"] = {
        "301": 89997,
        "651": 64623,
        "351": 105106,
        "401": 109292,
        "601": 72857,
    }
    return data_size_dict


def main():
    all_data_size_d = get_all_data_size()
    data_name = DataName("first_512_desc")

    target_data = ["351", "401", "601", "651"]
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