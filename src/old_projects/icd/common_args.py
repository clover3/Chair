import argparse

parser = argparse.ArgumentParser(description='File should be stored in ')

parser.add_argument("--batch_size", default=16)
parser.add_argument("--epochs", default=1)
parser.add_argument("--lr", default=1e-4)
