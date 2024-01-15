

from pyserini.index.lucene import IndexReader

from trainer_v2.chair_logging import c_log

index_reader = IndexReader.from_prebuilt_index('msmarco-v1-passage')


df_d = {}
for term in index_reader.terms():
    df_d[term.term] = term.df


c_log.info(f"Read df for {len(df_d)} terms")


#
# term = "city"
# print("Get posting list")
# posting_list = index_reader.get_postings_list(term)
#
# print("Traveling posting")
for posting in posting_list:
    print(f"doc_id: {posting.docid}, tf={posting.tf}")
    break
