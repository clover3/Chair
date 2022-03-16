import spacy


def main():
    nlp = spacy.load("en_core_web_sm")
    sent2 = "Where does real phrase origin phrase come from?"

    parsed = nlp(sent2)
    ch_idx_to_token_idx = {}
    for t_idx, t in enumerate(parsed):
        ch_idx_to_token_idx[t.idx] = t_idx

    for token in parsed:
        print(token, [ch_idx_to_token_idx[child.idx] for child in token.subtree])


if __name__ == "__main__":
    main()