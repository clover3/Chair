import time

from dataset_specific.msmarco.common import load_token_d_corpuswise_from_tsv


def main():
    split = "train"

    for job_id in range(367):
        st = time.time()
        tokens_d = load_token_d_corpuswise_from_tsv(split, job_id)
        ed = time.time()
        print("time per job", ed-st)


if __name__ == "__main__":
    main()