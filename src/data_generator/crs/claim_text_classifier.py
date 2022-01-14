import os
import pickle
import random

from elasticsearch import Elasticsearch

import cpath
from crs.load_stance_annotation import load_stance_verify_annot
from data_generator.tokenizer_wo_tf import EncoderUnitOld
from elastic import retrieve


class DataGenerator:
    def __init__(self):
        self.es = None
        name = "-1.csv"
        self.file_path= os.path.join(cpath.data_path, "crs", "verify", name)
        data = self.load_data(self.file_path)
        self.split_train_dev(data)
        vocab_filename = "bert_voca.txt"
        voca_path = os.path.join(cpath.data_path, vocab_filename)
        self.lower_case = True
        self.encoder = EncoderUnitOld(256, voca_path)

    def split_train_dev(self, data):
        group_by_statement = {}

        for e in data:
            key =e['statement']
            if key not in group_by_statement:
                group_by_statement[key] = []
            group_by_statement[key].append(e)

        n_group = len(group_by_statement)
        train_len = int(n_group * 0.8)
        dev_len = n_group - train_len
        self.train_split = []
        self.dev_split = []
        keys = list(group_by_statement.keys())
        random.shuffle(keys)
        train_keys = keys[:train_len]
        dev_keys = keys[train_len:]

        for k in train_keys:
            self.train_split.extend(group_by_statement[k])

        for k in dev_keys:
            self.dev_split.extend(group_by_statement[k])

    def load_data(self, file_path):
        annotation_data = self.merge_by_percent(file_path)
        text_dict = self.get_text_data(file_path, annotation_data)
        for e in annotation_data:
            e['text'] = text_dict[e['link']]
        return annotation_data

    def encode(self, e):
        text1 = e['text']
        text2 = e['statement']
        y0 = e['s_percent']
        y1 = e['d_percent']
        y0_sum = e['s_sum']
        y1_sum = e['d_sum']

        entry = self.encoder.encode_pair(text1, text2)
        return entry["input_ids"], entry["input_mask"], entry["segment_ids"], y0, y1, y0_sum, y1_sum

    def get_train_data(self):
        return list([self.encode(e) for e in self.train_split])

    def get_dev_data(self):
        return list([self.encode(e) for e in self.dev_split])

    def get_text_data(self, file_path, annotation_data):
        cache_path = file_path + ".txt.pickle"
        if os.path.exists(cache_path):
            return pickle.load(open(cache_path, "rb"))

        links = [e['link'] for e in annotation_data]
        texts = self.fetch_text(links)
        text_dict = dict(zip(links, texts))

        pickle.dump(text_dict, open(cache_path, "wb"))
        return text_dict

    def fetch_text(self, links):
        if self.es is None:
            server_name = "gosford.cs.umass.edu"
            self.es = Elasticsearch(server_name)

        text_list = []
        for link in links:
            tokens = link.split("/")
            doc_id = tokens[-1].replace("_", "/")
            seg_id = int(tokens[-2])
            if seg_id < 500:
                text = retrieve.get_paragraph(doc_id, seg_id)
            else:
                text = retrieve.get_comment(doc_id, seg_id)
            text_list.append(text)
        return text_list

    def merge_by_percent(self, path):
        data = load_stance_verify_annot(path)
        group = {}
        sig2data = {}
        for e in data:
            sig = e['statement'] + e['link']
            sig2data[sig] = e['statement'], e['link']
            if sig not in group:
                group[sig] = []

            group[sig].append((e['support'], e['dispute']))

        NOT_FOUND = 0
        YES = 1
        NOT_SURE = 2

        result_data = []
        for sig in group:
            statement, link = sig2data[sig]

            s_yes_cnt = 0
            s_no_cnt = 0
            d_yes_cnt = 0
            d_no_cnt = 0
            for s, d in group[sig]:
                if s == YES:
                    s_yes_cnt += 1
                elif s == NOT_FOUND:
                    s_no_cnt += 1

                if d == YES:
                    d_yes_cnt += 1
                elif d == NOT_FOUND:
                    d_no_cnt += 1

            s_sum = s_yes_cnt+s_no_cnt
            if s_sum != 0 :
                s_percent = s_yes_cnt / (s_yes_cnt+s_no_cnt)
            else:
                s_percent = 0

            d_sum = d_yes_cnt + d_no_cnt
            if d_sum != 0 :
                d_percent = d_yes_cnt / (d_yes_cnt+d_no_cnt)
            else:
                d_percent = 0

            result_data.append({
                'statement':statement,
                'link':link,
                's_percent':s_percent,
                's_sum':s_sum,
                'd_percent':d_percent,
                'd_sum':d_sum
            })

        return result_data

def data_stat():
    dg = DataGenerator()
    y0_list =[]
    y1_list =[]
    yc_list = []
    for e in dg.train_split:
        if e['s_sum'] > 0:
            y0 = e['s_percent'] > 0.5
            y0_list.append(y0)
        if e['d_sum'] > 0:
            y1 = e['d_percent'] > 0.5
            y1_list.append(y1)

    print("Support : ", sum(y0_list), len(y0_list))
    print("Disptue : ", sum(y1_list), len(y1_list))


def dev():
    dg = DataGenerator()
    dg.get_train_data()


if __name__ == "__main__":
    data_stat()



