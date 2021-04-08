from nltk.tokenize import sent_tokenize

from arg.counter_arg_retrieval.f5.load_f5_clue_docs import load_f5_docs_texts


def enum_data() :
    texts = load_f5_docs_texts()
    for text in texts:
        print("< Long >")
        print(text)
        for sent in sent_tokenize(text):
            print("< Short >")
            print(sent)


if __name__ == "__main__":
    enum_data()