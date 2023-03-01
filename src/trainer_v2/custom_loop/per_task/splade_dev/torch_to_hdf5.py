import sys
import torch

import h5py

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model = torch.load(input_path, map_location=torch.device('cpu'))

    fs = h5py.File(output_path, 'w')

    for key in model.keys():
        tensor = model[key]
        fs.create_dataset(key, data=tensor.numpy())

    fs.close()


if __name__ == "__main__":
    main()