import csv
from data_generator.common import *
corpus_dir = os.path.join(data_path, "stance_detection")


class DataLoader():
    def class_labels(self):
        return ["NONE", "AGAINST", "FAVOR"]

    def example_generator(self, corpus_path, select_target):
        label_list = self.class_labels()
        f = open(corpus_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.reader(f, delimiter=',')

        for idx, row in enumerate(reader):
            if idx == 0: continue  # skip header
            # Works for both splits even though dev has some extra human labels.
            sent = row[0]
            target = row[1]
            label = label_list.index(row[2])
            if target in select_target:
                yield {
                    "inputs": sent,
                    "label": label
                }



    def load_data(self):
        path = os.path.join(corpus_dir, "train.csv")
        data = self.example_generator(path, "atheism")