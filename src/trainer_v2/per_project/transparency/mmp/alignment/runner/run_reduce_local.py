import sys
from collections import Counter

from trainer_v2.per_project.transparency.mmp.alignment.reduce_local_to_global import sum_summarize_local_aligns_paired
from trainer_v2.per_project.transparency.mmp.when_corpus_based.runner.when_global_align import when_raw_tf


def main():
    input_file_path = sys.argv[1]
    save_path = sys.argv[2]
    corpus_tf: Counter = when_raw_tf()
    sum_summarize_local_aligns_paired(corpus_tf, input_file_path, save_path)


if __name__ == "__main__":
    main()