import sys

from explain.runner.eval_accuracy import eval_accuracy

if __name__  == "__main__":
    eval_accuracy(sys.argv[1], sys.argv[2], "snli")