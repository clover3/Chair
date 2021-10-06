import spacy


def main():
    old_text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."
    text = "In the treatment of patients with hypertension and renal-artery stenosis, angioplasty has little advantage over antihypertensive-drug therapy."
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    for token in doc:
        print(token.text, token.has_vector, token.vector_norm, token.is_oov)


if __name__ == "__main__":
    main()