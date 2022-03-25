from cache import load_list_from_jsonl
from tlm.tlm.runner.analyze_param import BinCounter
from trainer_v2.epr.path_helper import get_segmented_data_path


def main():
    file_path = get_segmented_data_path("snli", split="validation")
    dataset = load_list_from_jsonl(file_path, lambda x: x)
    ranges = [
        (0, 8),
        (8, 16),
        (16, 20),
        (20, 100),
        (100, 200),
        (200, 9000)
    ]
    counter = BinCounter(ranges)
    for e in dataset:
        n_words = len(e['hypothesis'])
        counter.add(n_words)

    for k, v in counter.count.items():
        print(k, v)



if __name__ == "__main__":
    main()