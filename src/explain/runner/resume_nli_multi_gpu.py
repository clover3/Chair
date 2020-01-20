import sys

from explain.runner.run_nli_on_multi_gpu import train_nil_from

if __name__  == "__main__":
    train_nil_from(sys.argv[1], sys.argv[2], True)