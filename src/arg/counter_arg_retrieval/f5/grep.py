from arg.counter_arg_retrieval.f5.f5_docs_payload_gen import enum_f5_data
from arg.counter_arg_retrieval.f5.load_f5_clue_docs import load_all_docs_cleaned


def main():
    rlp = "C:\\work\\Code\\Chair\\output\\clue_counter_arg\\ranked_list.txt"
    html_dir = "C:\\work\\Code\\Chair\\output\\clue_counter_arg\\docs"
    grouped = load_all_docs_cleaned(rlp, html_dir)

    for query, entries in grouped.items():
        for doc_id, text in entries:
            if "Why don't you re-write the example of the attack on the sender?" in text:
                print(doc_id)
                print(text)


def main2():
    for e in enum_f5_data():
        if "attack on the" in e:
            print(e)

if __name__ == "__main__":
    main2()