from data_generator.tokenizer_wo_tf import pretty_tokens
from datastore.interface import load
from datastore.table_names import BertTokenizedCluewebDoc


def main():
    doc_id = "clueweb12-0005wb-96-30750"
    doc = load(BertTokenizedCluewebDoc, doc_id)
    print("doc has {} lines", len(doc))
    print("last line:", pretty_tokens(doc[-1], True))



if __name__ == "__main__":
    main()