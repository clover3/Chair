from data_generator.data_parser import trec
from tlm.doc_length.sumarize_doc_length import summarize_doc_length


def main():
    robust_path = "/mnt/nfs/work3/youngwookim/data/robust04"
    data = trec.load_robust(robust_path)

    def iter_docs():
        for doc_id, text in data.items():
            yield text

    summarize_doc_length(iter_docs())


if __name__ == "__main__":
    main()