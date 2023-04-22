from cache import load_from_pickle, save_to_pickle
from dataset_specific.msmarco.passage.passage_resource_loader import enum_when_corpus_tokenized
from misc_lib import TELI


def load_qids():
    return load_from_pickle("when0_qids")


def load_when0_corpus():
    return load_from_pickle("when0_tokenized")


def build_tokenized():
    qids = load_qids()
    print("load {} qids".format(len(qids)))
    output = []
    qid_seen = set()
    last_qid = "-1"
    terminate_on_next_qid = False
    print(qids)
    for t in enum_when_corpus_tokenized():
        qid = t[0]
        if terminate_on_next_qid:
            if last_qid != qid:
                break

        if qid in qids:
            qid_seen.add(qid)
            print(len(qid_seen))
            output.append(t)

        if len(qid_seen) == len(qids):
            terminate_on_next_qid = True

        if last_qid != qid:
            last_qid = qid
            print(qid)


    save_to_pickle(output, "when0_tokenized")


def main():
    build_tokenized()
    return NotImplemented


if __name__ == "__main__":
    main()