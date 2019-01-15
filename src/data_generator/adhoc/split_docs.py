import data_generator.data_parser.trec as trec
import random


def split_docs():
    print("loading...")
    collection = trec.load_robust(trec.robust_path)
    window_size = 200 * 3

    def sample_shift():
        return random.randrange(0, window_size * 2)

    fout = open("rob04.split.txt", "w")
    def write(new_id, text_span):
        fout.write("<DOC>\n")
        fout.write("<DOCNO>{}</DOCNO>\n".format(new_id))
        fout.write("<TEXT>\n")
        fout.write(text_span)
        fout.write("</TEXT>\n")
        fout.write("</DOC>\n")

    print("writing...")
    for doc_id in collection:
        content = collection[doc_id]
        loc_ptr = 0
        while loc_ptr < len(content):
            text_span = content[loc_ptr:loc_ptr + window_size]
            new_id = doc_id + "_{}".format(loc_ptr)
            write(new_id, text_span)
            loc_ptr += sample_shift()


if __name__ == "__main__":
    split_docs()

