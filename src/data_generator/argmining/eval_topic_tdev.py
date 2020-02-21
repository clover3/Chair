import sys

from data_generator.argmining.eval_topic import eval_topic

if __name__ == "__main__":
    eval_topic(sys.argv[1], sys.argv[2], "tdev", "tdev")