from adhoc.build_index import save_inv_index_to_pickle
from dataset_specific.beir_eval.beir_common import load_beir_dataset, beir_dataset_list_A, avdl_luk
from dataset_specific.msmarco.passage.runner.lucene_corpus_inv_index import build_inverted_index
from trainer_v2.chair_logging import c_log
from pyserini.analysis import Analyzer, get_lucene_analyzer
from cpath import output_path
from misc_lib import path_join, exist_or_mkdir
import sys
from omegaconf import OmegaConf


def build_beir_luk_conf(dataset_name):
    dir_path = path_join(output_path, "beir_luk", dataset_name)
    exist_or_mkdir(dir_path)
    conf = OmegaConf.create({
        "index_name": "lucene_krovetz_" + dataset_name,
        "tokenizer": "lucene_krovetz",
        'inv_index_path': path_join(dir_path, "inv_index.pkl"),
        'bg_prob_path': path_join(dir_path, "bg_prob.pkl"),
        'df_path': path_join(dir_path, "df.pkl"),
        'dl_path': path_join(dir_path, "dl.pkl"),
        'avdl': avdl_luk[dataset_name],
        "b": 0.4,
        "k1": 0.9,
        "k2": 100
    })
    return conf


def main():
    analyzer = Analyzer(get_lucene_analyzer(stemmer='krovetz'))
    tokenize_fn = analyzer.analyze
    dataset = sys.argv[1]
    conf = build_beir_luk_conf(dataset)
    corpus, queries, qrels = load_beir_dataset(dataset, "test")

    def iter_tokenized():
        for doc_id, doc in corpus.items():
            doc_text = doc['title'] + " " + doc['text']
            yield doc_id, tokenize_fn(doc_text)

    corpus_tokenized = iter_tokenized()
    collection_size = len(corpus)

    c_log.info(f"Working on {dataset}")
    outputs = build_inverted_index(
        corpus_tokenized,
        collection_size,
    )
    save_inv_index_to_pickle(conf, outputs)


if __name__ == "__main__":
    main()
