import spacy

from cache import save_to_pickle, load_from_pickle


class TestObj:

    def __init__(self):
        self.d = {}


    def __del__(self):
        save_to_pickle(self.d, "spacy_test_pickle2")

def save_spacy():
    nlp = spacy.load("en_core_web_sm")
    text1 = "There is still a place for mercenaries working for NGOs."
    text2 = "Humanitarian mercenaries"
    text3 = "Legislation against mercenaries"

    d1 = nlp(text1)
    d2 = nlp(text2)

    t = TestObj()
    t.d = {'text1': d1, 'text2': d2}
    return t

def load_spacy():
    nlp = spacy.load("en_core_web_sm")

    d = load_from_pickle("spacy_test_pickle2")
    print(d)


def main():
    t1 = save_spacy()
    load_spacy()
    t2 = save_spacy()
    load_spacy()
    t3 = save_spacy()
    load_spacy()
    print(t1)


if __name__ == "__main__":
    main()
