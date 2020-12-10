from cache import save_to_pickle
from data_generator.data_parser.robust2 import load_bm25_best
from data_generator.data_parser.trec import load_robust


def main():
    top_k = 1000
    galago_rank = load_bm25_best()

    doc_id_set = set()
    for query_id, ranked_list in galago_rank.items():
        ranked_list.sort(key=lambda x :x[1])
        doc_id_set.update([x[0] for x in ranked_list[:top_k]])
    doc_id_list = list(doc_id_set)
    robust_path = "/mnt/nfs/work3/youngwookim/data/robust04"
    data = load_robust(robust_path)

    save_d = {}
    for doc_id in doc_id_list:
        try:
            save_d[doc_id] = data[doc_id]
        except KeyError:
            print(doc_id, 'not found')


    save_to_pickle(save_d, "robust04_docs_predict")



if __name__ == "__main__":
        main()