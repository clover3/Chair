
import sys

from evals.parse import load_qrels_flat
from trec.trec_parse import write_trec_relevance_judgement, TrecRelevanceJudgementEntry


def main():
    judgment_path = sys.argv[1]
    save_path = sys.argv[2]
    # print
    qrels = load_qrels_flat(judgment_path)

    def iter():
        for query_id, docs in qrels.items():
            for doc_id, raw_score in docs:
                if raw_score > 0:
                    score = raw_score
                else:
                    score = 0
                yield TrecRelevanceJudgementEntry(query_id, doc_id, score)

    write_trec_relevance_judgement(iter(), save_path)


if __name__ == "__main__":
    main()
