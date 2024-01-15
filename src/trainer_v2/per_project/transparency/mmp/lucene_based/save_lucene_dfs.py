import pickle

from pyserini.index.lucene import IndexReader

from trainer_v2.chair_logging import c_log
from cpath import output_path
from misc_lib import path_join


index_reader = IndexReader.from_prebuilt_index('msmarco-v1-passage')


df_d = {}
for term in index_reader.terms():
    df_d[term.term] = term.df


c_log.info(f"Read df for {len(df_d)} terms")

save_path = path_join(output_path, "mmp", "lucene_dfs")
pickle.dump(df_d, open(save_path, "wb"))


