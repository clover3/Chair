
import os
from path import data_path
import csv
import nltk

scope_path = os.path.join(data_path, "arg", "extend")
sentence_path = os.path.join(scope_path, "sentences")
out_dir = os.path.join(scope_path, "sentences_filtered")

def filter_document(in_path, out_path):
    def has_verb(pos_pair_list):
        for word, pos in pos_pair_list:
            if "VB" in pos:
                return True
        return False

    f_out = csv.writer(open(out_path, "w"), dialect='excel-tab')
    f = open(in_path, "r")
    cnt = 0
    column = None
    try:
        for row in csv.reader(f, dialect='excel-tab', quoting=csv.QUOTE_NONE):
            if cnt == 0:
                column = row
                idx_sentence = column.index('sentence')
                f_out.writerow(row)
            else:
                sentence = row[idx_sentence]
                tokens = sentence.split()
                if len(tokens) >= 3:
                    tags = nltk.pos_tag(tokens)
                    if has_verb(tags):
                        f_out.writerow(row)

            cnt += 1
    except Exception as e:
        print(e)
        raise e

def filter_sents():
    docs = []
    for (dirpath, dirnames, filenames) in os.walk(sentence_path):
        docs.extend(filenames)


    for name in docs:
        in_path = os.path.join(sentence_path, name)
        out_path = os.path.join(out_dir, name)
        print(out_path)
        filter_document(in_path, out_path)

if __name__ == "__main__":
    filter_sents()
