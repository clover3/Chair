from list_lib import flatten
from trec.trec_parse import write_trec_relevance_judgement
from trec.types import load_qrel_as_entries


def main():
    qrel_path = "C:\\work\\Code\\Chair\\output\\ca_building\\qrel\\0522.txt"
    qrel_save_path = "C:\\work\\Code\\Chair\\output\\ca_building\\qrel\\0522_done.txt"
    qrel = load_qrel_as_entries(qrel_path)
    qrel_done = {qid: entries for qid, entries in qrel.items() if len(entries) > 18}
    print("{} of {} queries are done".format(len(qrel_done), len(qrel)))
    write_trec_relevance_judgement(flatten(qrel_done.values()), qrel_save_path)


if __name__ == "__main__":
    main()