import os
from path import data_path


agree_path = os.path.join(data_path,"agree", "commentAgree.txt")

# 0 : neutral
# 1 : agree
# 2 : disagree
def load_agree_data():
    for line in open(agree_path, encoding="utf-8"):
        label = int(line[:1])
        content = line[2:]
        yield (content, label)


