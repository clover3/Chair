import spacy

from contradiction.medical_claims.annotation_1.load_data import load_dev_pairs
from models.classic.stopword import load_stopwords


def main():
    output = load_dev_pairs()
    nlp = spacy.load("en_core_web_lg")
    stopwords = load_stopwords()
    for group_no, pairs in output:
        for pair in pairs:
            oov = []
            text1, text2 = pair
            print(text1)
            print(text2)
            nlp_text1 = nlp(text1)
            nlp_text2 = nlp(text2)
            for token2 in nlp_text2:
                if token2.is_oov:
                    oov.append(token2)
                    continue

                tokens1 = list(nlp_text1)
                tokens1.sort(key=token2.similarity, reverse=True)
                token1 = tokens1[0]
                print(token2, token1, token2.similarity(token1))
            print(oov)
            break
        break


if __name__ == "__main__":
    main()