import pickle
import sys


def main():
    save_path = sys.argv[1]
    obj = pickle.load(open(save_path, "rb"))

    batch0 = obj[0]
    for key in batch0:
        if key in ["input_ids", "label_ids"]:
            continue
        print(key)
        print(batch0[key])


if __name__ == "__main__":
    main()