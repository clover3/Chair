from cache import save_to_pickle
from clueweb.adhoc.load_docs import read_doc_id_title_text
from cpath import at_output_dir


def main():
    f = open(at_output_dir("clueweb", "doc_ids_sample.txt"), "r")
    doc_ids = list([l.strip() for l in f])

    doc_contents = read_doc_id_title_text()

    new_d = {}
    for doc_id in doc_ids:
        t = doc_contents[doc_id]
        new_d[doc_id] = t

    save_to_pickle(new_d, "clean_clueweb_doc_sample")


if __name__ == "__main__":
    main()

