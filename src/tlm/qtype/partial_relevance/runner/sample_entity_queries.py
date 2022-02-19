import spacy

from dataset_specific.msmarco.common import load_queries


def main():
    split = "dev"
    queries = load_queries(split)
    nlp = spacy.load("en_core_web_sm")
    not_use = ["CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"]
    use = ["EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"]

    for qid, query  in queries:
        doc = nlp(query)
        valid_entity_list = []
        if doc.ents:
            for e in doc.ents:
                if e.label_ in use:
                    valid_entity_list.append(e)
        if valid_entity_list:
            print("query:", query)
            for e in valid_entity_list:
                print(e, e.label_)
            print("--")


if __name__ == "__main__":
    main()
