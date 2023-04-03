from collections import defaultdict, Counter

from adhoc.bm25_class import BM25
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from cpath import output_path
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter, load_qrel
from misc_lib import path_join, get_second, two_digit_float, pause_hook
from trainer_v2.chair_logging import c_log


def get_data_iter():
    source_corpus_path = path_join("data", "msmarco", "passage", "grouped_10K", "0")
    return tsv_iter(source_corpus_path)


def main():
    c_log.info("Loading term stats")
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25)

    itr = get_data_iter()
    per_query_dict = defaultdict(list)
    n_record = 0
    try:
        for e in itr:
            qid, pid, _, _ = e
            n_record += 1
            l = per_query_dict[qid]
            l.append(e)
    except Exception as e:
        print(e)

    c_log.info("Loading Qrels")
    qrel = load_qrel("train")

    def enum_valid_entries():
        for qid, items in per_query_dict.items():
            if len(items) < 20:
                c_log.info("Skip because there are not many passages available")
                continue
            if qid not in qrel:
                c_log.info("Skip because there is no qrel")
                continue

            pid_list = [pid for qid, pid, _, _  in items]

            pos_doc_ids = enum_pos_doc_ids_for(qid)
            if not pos_doc_ids:
                continue

            if pos_doc_ids[0] not in pid_list:
                c_log.info("Skip relevant document is not in ranked list")
                continue

            yield qid, items

    def enum_pos_doc_ids_for(qid):
        pos_doc_ids = []
        for doc_id, score in qrel[qid].items():
            if score > 0:
                pos_doc_ids.append(doc_id)
        return pos_doc_ids

    tokenizer = KrovetzNLTKTokenizer()
    for qid, per_term_info in pause_hook(enum_valid_entries(), 10):
        _, _, query_text, _ = per_term_info[0]

        q_tokens = tokenizer.tokenize_stem(query_text)

        score_list = []
        pid_to_doc = {}
        for qid, pid, _, doc in per_term_info:
            score = bm25.score(query_text, doc)
            score_list.append((pid, score))
            pid_to_doc[pid] = doc

        score_list.sort(key=get_second, reverse=True)
        pos_doc_id = enum_pos_doc_ids_for(qid)[0]
        print()
        print("Query: {} : {}".format(query_text, q_tokens))

        not_found_counter = Counter()
        for pid, score in score_list:
            rel = pid == pos_doc_id
            rel_str = "Relevant" if rel else "NotRelev"
            # This is a false positive document
            doc = pid_to_doc[pid]
            doc_tf = Counter(tokenizer.tokenize_stem(doc))
            all_match = True
            per_term_info = []
            for q_term, q_tf in Counter(q_tokens).items():
                d_tf = doc_tf[q_term]
                if d_tf == 0:
                    all_match = False
                # s = bm25.score_inner({q_term: q_tf}, {q_term: d_tf})
                if d_tf == 0:
                    not_found_counter[q_term] += 1
                out_str = str(d_tf > 0)
                # out_str = f"{q_term} ({s:.2f})"
                per_term_info.append(out_str)

            summary = "{} all_match={} Score={} {}".format(
                rel_str,
                all_match,
                two_digit_float(score),
                " ".join(per_term_info))

            # print(summary)
            if pid == pos_doc_id:
                break

        print("Not found:", not_found_counter)




if __name__ == "__main__":
    main()