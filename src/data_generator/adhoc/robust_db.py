import data_generator.data_parser.trec as trec
from nltk.tokenize import sent_tokenize
from tlm.mysql_sentence import add_sents

def add_rob_sents():
    print("loading...")
    collection = trec.load_robust(trec.robust_path)
    print("writing...")
    for doc_id in collection:
        content = collection[doc_id]
        sents = sent_tokenize(content)


        for i, s in enumerate(sents):
            if len(s) > 500:
                sents[i] = s[:500]

        add_sents(doc_id, sents)


if __name__ == "__main__":
    add_rob_sents()

