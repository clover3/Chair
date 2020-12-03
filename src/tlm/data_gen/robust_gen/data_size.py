from typing import Dict


def get_all_data_size() -> Dict[str, Dict]:
    all_data_size_d: Dict[str, Dict] = dict()
    all_data_size_d["robust_K"] = {
        "651": 222338,
        "301": 467338,
        "601": 358720,
        "401": 821393,
        "351": 610591
    }
    all_data_size_d["robust_L"] = {
        "401": 1483830,
        "351": 1050250,
        "601": 597750,
        "301": 757794,
        "651": 279430,
    }
    return all_data_size_d


def main():
    all_data_size_d = get_all_data_size()
    data_name = "robust_L"

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