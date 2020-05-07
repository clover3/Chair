import collections
import os

import data_generator.argmining.ukp_header
from cpath import data_path
from data_generator import job_runner
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.job_runner import JobRunner, sydney_working_dir
from data_generator.tokenizer_wo_tf import pretty_tokens, get_tokenizer
from list_lib import lmap, flatten
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import get_basic_input_feature_as_list, \
    log_print_feature
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take
from tlm.tlm.relevance_on_bert_tokens import Ranker
from tlm.ukp.data_gen.add_topic_ids_cls import token_ids_to_topic
from tlm.ukp.data_gen.run_ukp_gen2 import load_tokens_for_topic
from tlm.ukp.sydney_data import sydney_get_ukp_ranked_list


def load_train_data(input_path):
    for feature in load_record(input_path):
        input_ids = feature["input_ids"].int64_list.value
        label_ids = feature["label_ids"].int64_list.value[0]
        yield input_ids, label_ids



def pool_tokens(sent_list, target_seq_length, overlap_tokens=200):
    results = []
    current_chunk = []
    current_length = 0
    prev_tokens = []
    i = 0
    while i < len(sent_list):
        segment = sent_list[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(sent_list) - 1 or current_length >= target_seq_length:
            tokens_a = prev_tokens + flatten(current_chunk)
            tokens_a = tokens_a[:target_seq_length]
            results.append(tokens_a)
            current_chunk = []
            prev_tokens = tokens_a[-overlap_tokens:]
            current_length = len(prev_tokens)
        i += 1

    return results


def filter_overlap(ranked_list):
    for i in range(1, len(ranked_list)):
        _, score, tf = ranked_list[i-1]
        _, score2, tf2 = ranked_list[i]

        skip = False
        if abs(score-score2) < 0.001:
            diff = False
            for key in tf:
                if tf[key] != tf2[key]:
                    diff = True
                    break
            if not diff:
                skip = True
        if not skip:
            yield ranked_list[i]


n_topics = len(data_generator.argmining.ukp_header.all_topics)

def get_tf_record_path_from_topic(split, topic):
    return os.path.join(data_path, "ukp_tfrecord", "{}_{}".format(split, topic))


class ContextGenWorker(job_runner.WorkerInterface):
    def __init__(self, split, out_path):
        self.out_dir = out_path
        self.tokenizer = get_tokenizer()
        self.max_seq_length = 128
        self.max_context_len = 512 - 128
        self.max_context = 16
        self.split = split


    def work(self, job_id):
        idx1 = int(job_id / n_topics)
        idx2 = job_id % n_topics

        heldout_topic = data_generator.argmining.ukp_header.all_topics[idx1]
        target_topic = data_generator.argmining.ukp_header.all_topics[idx2]
        input_path = get_tf_record_path_from_topic(self.split, heldout_topic)
        instances = self.create_instances(input_path, target_topic, self.max_seq_length)
        out_name = "{}_{}_{}".format(self.split, heldout_topic, target_topic)
        self.write_instance(instances, os.path.join(self.out_dir, out_name))

    def create_instances(self, input_path, target_topic, target_seq_length):
        tokenizer = get_tokenizer()
        doc_top_k = 1000

        all_train_data = list(load_record(input_path))
        train_data = []
        for feature in all_train_data:
            input_ids = feature["input_ids"].int64_list.value
            token_id = input_ids[1]
            topic = token_ids_to_topic[token_id]
            if target_topic == topic:
                train_data.append(feature)

        print("Selected {} from {}".format(len(train_data), len(all_train_data)))

        doc_dict = load_tokens_for_topic(target_topic)
        token_doc_list = []
        ranked_list = sydney_get_ukp_ranked_list()[target_topic]
        print("Ranked list contains {} docs, selecting top-{}".format(len(ranked_list), doc_top_k))
        doc_ids = [doc_id for doc_id, _, _ in ranked_list[:doc_top_k]]

        for doc_id in doc_ids:
            doc = doc_dict[doc_id]
            token_doc = pool_tokens(doc, target_seq_length)
            token_doc_list.extend(token_doc)

        ranker = Ranker()
        target_tf_list = lmap(ranker.get_terms, token_doc_list)

        ranker.init_df_from_tf_list(target_tf_list)

        inv_index = collections.defaultdict(list)
        for doc_idx, doc_tf in enumerate(target_tf_list):
            for term in doc_tf:
                if ranker.df[term] < ranker.N * 0.3:
                    inv_index[term].append(doc_idx)


        def get_candidate_from_inv_index(inv_index, terms):
            s = set()
            for t in terms:
                s.update(inv_index[t])
            return s

        source_tf_list = []
        selected_context = []
        for s_idx, feature in enumerate(train_data):
            input_ids = feature["input_ids"].int64_list.value
            topic_seg, sent = split_p_h_with_input_ids(input_ids, input_ids)
            source_tf = ranker.get_terms_from_ids(sent)
            source_tf_list.append(source_tf)
            ranked_list = []
            candidate_docs = get_candidate_from_inv_index(inv_index, source_tf.keys())
            for doc_idx in candidate_docs:
                target_tf = target_tf_list[doc_idx]
                score = ranker.bm25(source_tf, target_tf)
                ranked_list.append((doc_idx, score, target_tf))
            ranked_list.sort(key=lambda x: x[1], reverse=True)
            ranked_list = list(filter_overlap(ranked_list))
            ranked_list = ranked_list[:self.max_context]

            if s_idx < 10:
                print("--- Source sentence : \n", pretty_tokens(tokenizer.convert_ids_to_tokens(sent), True))
                print("-------------------")
                for rank, (idx, score, target_tf) in enumerate(ranked_list):
                    ranker.bm25(source_tf, target_tf, True)
                    print("Rank#{}  {} : ".format(rank, score) + pretty_tokens(token_doc_list[idx], True))
            if s_idx % 100 == 0:
                print(s_idx)
            contexts = list([token_doc_list[idx] for idx, score, _ in ranked_list])
            selected_context.append(contexts)

        for sent_idx, feature in enumerate(train_data):
            contexts = selected_context[sent_idx]
            yield feature, contexts

    def write_instance(self, instances, output_path):
        writer = RecordWriterWrap(output_path)
        for (inst_index, instance) in enumerate(instances):
            new_features = collections.OrderedDict()
            feature, contexts = instance
            for key in feature:
                v = take(feature[key])
                new_features[key] = create_int_feature(v[:self.max_seq_length])

            context_input_ids = []
            context_input_mask = []
            context_segment_ids = []

            for tokens in contexts:
                segment_ids = [0] * len(tokens)
                input_ids, input_mask, segment_ids = \
                    get_basic_input_feature_as_list(self.tokenizer, self.max_context_len, tokens, segment_ids)
                context_input_ids.extend(input_ids)
                context_input_mask.extend(input_mask)
                context_segment_ids.extend(segment_ids)

            dummy_len = self.max_context - len(contexts)
            for _ in range(dummy_len):
                input_ids, input_mask, segment_ids = \
                    get_basic_input_feature_as_list(self.tokenizer, self.max_context_len, [], [])
                context_input_ids.extend(input_ids)
                context_input_mask.extend(input_mask)
                context_segment_ids.extend(segment_ids)

            new_features["context_input_ids"] = create_int_feature(context_input_ids)
            new_features["context_input_mask"] = create_int_feature(context_input_mask)
            new_features["context_segment_ids"] = create_int_feature(context_segment_ids)
            writer.write_feature(new_features)
            if inst_index < 20:
                log_print_feature(new_features)
        writer.close()



if __name__ == "__main__":
    num_jobs = n_topics * n_topics - 1
    JobRunner(sydney_working_dir, num_jobs, "context_ukp", lambda x: ContextGenWorker("train", x)).start()

