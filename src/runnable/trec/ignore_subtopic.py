
import sys

from evals.parse import load_qrels_with_subtopic
from trec.trec_parse import write_trec_relevance_judgement, TrecRelevanceJudgementEntry


def main():
    judgment_path = sys.argv[1]
    save_path = sys.argv[2]
    # print
    qrels = load_qrels_with_subtopic(judgment_path)

    new_d = {}
    for query_id, docs in qrels.items():
        if query_id not in new_d:
            new_d[query_id] = {}

        for doc_id, subtopic, raw_score in docs:
            if raw_score > 0:
                score = raw_score
                if doc_id in new_d[query_id]:
                    # other sub-topic is judged, which should be relevant.
                    assert new_d[query_id][doc_id] > 0

                new_d[query_id][doc_id] = raw_score
            else:
                # If not relevant, no sub-topic is relevant
                assert doc_id not in new_d[query_id]
                score = 0
                new_d[query_id][doc_id] = score


    def iter():
        for query_id, doc_score_d in new_d.items():
            for doc_id in doc_score_d:
                score = doc_score_d[doc_id]
                yield TrecRelevanceJudgementEntry(query_id, doc_id, score)

    write_trec_relevance_judgement(iter(), save_path)


if __name__ == "__main__":
    main()
