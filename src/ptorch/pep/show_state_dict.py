import sys
import torch


def main():
    file_path = sys.argv[1]
    d = torch.load(file_path)
    state_dict = d['model_state_dict']
    keys = list(state_dict.keys())
    print(keys[:10])


if __name__ == "__main__":
    main()