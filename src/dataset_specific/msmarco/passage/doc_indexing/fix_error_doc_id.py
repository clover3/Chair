from cpath import output_path
from dataset_specific.msmarco.passage_common import enum_passage_corpus
from misc_lib import path_join


def fix_error():
    work_path = path_join(output_path, "msmarco", "msmarco_passage_tokenize")
    num_item = 9

    def enum_tokenized():
        for i in range(num_item):
            file_path = path_join(work_path, str(i))
            for line in open(file_path, "r"):
                yield line

    save_path = path_join(output_path, "msmarco", "msmarco_passage_tokenize", "all")
    f = open(save_path, "w")
    cnt = 0
    itr1 = enum_passage_corpus()
    itr2 = enum_tokenized()
    for e1, e2 in zip(itr1, itr2):
        doc_id, doc_text = e1
        f.write("{}\t{}\n".format(doc_id, e2.strip()))
        cnt += 1

        if cnt % 1000000 == 0:
            print(doc_text, e2)

    f.close()



def main():
    fix_error()


if __name__ == "__main__":
    main()