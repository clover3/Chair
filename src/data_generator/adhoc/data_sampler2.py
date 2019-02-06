
import path
from data_generator.data_parser.trec import *
from misc_lib import *
from adhoc.bm25 import get_bm25
from data_generator.cached_tokenizer import EncoderUnit

class DataSampler:
    def __init__(self, queries, collection):
        self.collection = collection
        self.inv_index = get_inverted_index(self.collection)
        self.idf = Idf(list(self.collection.values()))
        self.threshold_boring_doc = 20
        self.min_posting = 5
        self.inst_per_query = 30
        self.queries = queries

    def save_to_pickle(self, pickle_name):
        save_to_pickle(self, pickle_name)

    @classmethod
    def init_from_pickle(cls, pickle_name):
        return load_from_pickle(pickle_name)

    def pair_generator(self):
        ranked_list = self.ranked_list_generate()
        for query, score_group in ranked_list:
            candidate = []
            for key_score, span_list in score_group.items():
                for score, span in span_list:
                    candidate.append((score, span))
            # print("candidate len : {}".format(len(candidate)))
            for i in range(self.inst_per_query):
                l = random.sample(range(len(candidate)), 2)
                x1 = candidate[l[0]]
                x2 = candidate[l[1]]
                # print(x1[0], x1[1][:100])
                # print(x2[0], x2[1][:100])
                if x1[0] < x2[0]:
                    yield query, x1, x2
                else:
                    yield query, x2, x1

    def tfidf_span(self, q_terms, text_span):
        return sum([text_span.count(q_i) * self.idf[q_i] for q_i in q_terms])

    def check_worthy(self, q_terms, doc_id_list):
        max_score = 0
        for doc_id in doc_id_list:
            raw_document = self.collection[doc_id]
            loc_ptr = 0
            while loc_ptr < len(raw_document):
                text_span = raw_document[loc_ptr:loc_ptr + self.window_size]

                score = self.tfidf_span(q_terms, text_span)
                max_score = max(score, max_score)
                loc_ptr += self.window_size
        return max_score >= self.threshold_boring_doc

    def ranked_list_generate(self):
        self.window_size = 512 * 3

        def flatten_and_get_doc_id(postings_list):
            doc_ids = []
            for postings in postings_list:
                for doc_id, idx in postings:
                    doc_ids.append(doc_id)
            return doc_ids

        def sample_shift():
            return random.randrange(0, self.window_size * 4)

        def sample_debiase(tf_n_span):
            max_occurence = 2

            output = {}
            for score, span in tf_n_span:
                score_grouper = int(score + 0.8)
                if score_grouper not in output:
                    output[score_grouper] = []
                if len(output[score_grouper]) < max_occurence:
                    output[score_grouper].append((score, span))
            return output

        collection_len = 252359881
        avdl = collection_len / len(self.collection)

        random.shuffle(self.queries)
        for query in self.queries:
            q_terms = query.split()
            postings_list = []
            for qterm in q_terms:
                postings = self.inv_index[qterm]
                if len(postings) < self.min_posting:
                    break  # Skip this query
                postings_list.append(postings)

            if not postings_list:
                continue
            print("Query :", query)
            doc_id_list = flatten_and_get_doc_id(postings_list)
            print("Docs : {}".format(len(doc_id_list)))

            spans = []
            if len(doc_id_list) > 1000:
                doc_id_list = random.sample(doc_id_list, 1000)

            if not self.check_worthy(q_terms, doc_id_list):
                continue

            # Scan docs and retrieve spans
            for doc_id in doc_id_list:
                raw_document = self.collection[doc_id]
                loc_ptr = sample_shift()
                while loc_ptr < len(raw_document):
                    text_span = raw_document[loc_ptr:loc_ptr + self.window_size]
                    score = get_bm25(" ".join(q_terms), text_span, self.idf.df, N=len(self.collection), avdl=avdl)
                    spans.append((score, text_span))
                    loc_ptr += sample_shift()
            score_group = sample_debiase(spans)
            for key in score_group:
                print(key)
            yield query, score_group


def save_data_samples():
    data_sampler = DataSampler.init_from_pickle("robust04")
    pair_generator = data_sampler.pair_generator()
    block_size = 1000
    for i in range(100):
        result = []
        while len(result) < block_size:
            raw_inst = pair_generator.__next__()
            result.append(raw_inst)

        pickle.dump(result, open("../output/plain512/{}.pickle".format(i), "wb"))

def encode(job_id):
    cache_path = os.path.join(path.cache_path, "sub_tokens.pickle")
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    encoder = EncoderUnit(cache_path=cache_path, max_sequence=512, voca_path=voca_path)
    result = pickle.load(open("../output/plain512/{}.pickle".format(job_id), "rb"))




if __name__ == '__main__':
    if sys.argv[1] == "encode":
        encode(int(sys.argv[2]))
    elif sys.argv[1] == "sample":
        save_data_samples()

