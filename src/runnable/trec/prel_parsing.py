from cpath import at_data_dir
from trec.trec_parse import TrecRelevanceJudgementEntry, write_trec_relevance_judgement


def main():
    print("using second score as judgments")
    input_path = at_data_dir("clueweb", "2009.prels.1-50")

    raw_entries = []
    for line in open(input_path, "r"):
        query_id, doc_id, s1, s2, s3 = line.split()
        maybe_relevance = int(s1)
        maybe_relevance2 = int(s2)
        some_float = float(s3)
        e = TrecRelevanceJudgementEntry(query_id, doc_id, maybe_relevance2)
        raw_entries.append(e)

    save_path = at_data_dir("clueweb", "2009.qrel_test.2.txt")
    write_trec_relevance_judgement(raw_entries, save_path)


if __name__ == "__main__":
    main()