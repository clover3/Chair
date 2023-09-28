from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_collection
from misc_lib import TELI


def main():
    itr = load_msmarco_collection()
    size = 8841823
    itr = TELI(itr, size)

    n_token_sum = 0
    n_doc = 0
    for _, doc in itr:
        n_token_sum += len(doc.split())
        n_doc += 1

    avdl = n_token_sum / n_doc
    # 56.25
    print(avdl)


if __name__ == "__main__":
    main()