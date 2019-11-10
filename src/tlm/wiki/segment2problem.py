import random
from cache import *
from path import data_path
from data_generator import tokenizer_wo_tf as tokenization
from sydney_manager import ReadyMarkTaskManager
from misc_lib import flatten
from tlm.retrieve_lm.retreive_candidates import get_visible
from tlm.stem import CacheStemmer
from tlm.retrieve_lm.galago_query_maker import clean_query
from collections import Counter
from adhoc.bm25 import BM25_3_q_weight
from misc_lib import left, TimeEstimator
from models.classic.stopword import load_stopwords
from adhoc.galago import load_df, write_query_json

working_path ="/mnt/nfs/work3/youngwookim/data/tlm_simple"

class ProblemMaker:
    def __init__(self):
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.stemmer = CacheStemmer()
        self.stopword = load_stopwords()
        self.df = self.load_galgo_df_stat()

    def load_galgo_df_stat(self):
        return load_df(os.path.join(data_path, "enwiki", "tf_stat"))

    def generate_mask(self, inst, max_num_tokens, masked_lm_prob, short_seq_prob, rng):
        max_predictions_per_seq = 20

        cur_seg, prev_seg, next_seg = inst

        def get_seg_tokens(seg):
            if seg is None:
                return None
            title, content, st, ed = seg
            return flatten([self.tokenizer.tokenize(t) for t in content])

        title, content, st, ed = cur_seg
        prev_tokens = get_seg_tokens(prev_seg)
        target_tokens = get_seg_tokens(cur_seg)
        next_tokens = get_seg_tokens(next_seg)

        if rng.random() < short_seq_prob and next_tokens is not None:
            target_seq_length = rng.randint(2, max_num_tokens)
            short_seg = target_tokens[:target_seq_length]
            remain_seg = target_tokens[target_seq_length:]
            next_tokens = (remain_seg + next_tokens)[:max_num_tokens]
            target_tokens = short_seg

        num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(target_tokens) * masked_lm_prob))))

        cand_indice = list(range(0, len(target_tokens)))
        rng.shuffle(cand_indice)
        mask_indice = cand_indice[:num_to_predict]
        doc_id = "{}-{}-{}".format(title, st,ed)
        mask_inst = target_tokens, prev_seg, cur_seg, next_seg, prev_tokens, next_tokens, mask_indice, doc_id
        return mask_inst

    def filter_stopword(self, l):
        return list([t for t in l if t not in self.stopword])

    def generate_query(self, mask_inst):
        target_tokens, prev_seg, cur_seg, next_seg, prev_tokens, next_tokens, mask_indice, doc_id = mask_inst

        title, content, st, ed = cur_seg

        visible_tokens = get_visible(content, target_tokens, mask_indice, self.tokenizer)

        stemmed = list([self.stemmer.stem(t) for t in visible_tokens])
        query = clean_query(self.filter_stopword(stemmed))
        high_qt_stmed = self.high_idf_q_terms(Counter(query))

        query = []
        for t in visible_tokens:
            if self.stemmer.stem(t) in high_qt_stmed:
                query.append(t)

        if not query:
            print(title)
            print(content)
            print(visible_tokens)
            print(high_qt_stmed)
        return query

    def high_idf_q_terms(self, q_tf, n_limit=10):
        total_doc =11503029 + 100

        high_qt = Counter()
        for term, qf in q_tf.items():
            qdf = self.df[term]
            w = BM25_3_q_weight(qf, qdf, total_doc)
            high_qt[term] = w

        return set(left(high_qt.most_common(n_limit)))

in_path_format = os.path.join(data_path, "stream_pickled", "wiki_segments3_{}")

def work(job_id, pm):
    rng = random.Random(0)
    max_num_tokens = 256
    masked_lm_prob = 0.15
    short_seq_prob = 0.1
    problem_per_job = 100*1000

    in_path = in_path_format.format(job_id)
    out_path = os.path.join(working_path, "problems", "{}".format(job_id))
    query_out_path = os.path.join(working_path, "query", "{}".format(job_id))
    in_data = pickle.load(open(in_path, "rb"))
    out_data = []
    queries = []

    ticker = TimeEstimator(len(in_data))

    for idx, inst in enumerate(in_data):
        mask_inst = pm.generate_mask(inst, max_num_tokens, masked_lm_prob, short_seq_prob, rng)
        query = pm.generate_query(mask_inst)
        qid = job_id * problem_per_job + idx
        queries.append((qid, query))
        out_data.append(mask_inst)
        ticker.tick()

    write_query_json(queries, query_out_path)
    pickle.dump(out_data, open(out_path, 'wb'))

def main():
    mark_path = os.path.join(working_path, "wiki_p2_mark")
    mtm = ReadyMarkTaskManager(1000, in_path_format, mark_path)

    pm = ProblemMaker()
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        work(job_id, pm)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)

if __name__ == "__main__":
    main()
