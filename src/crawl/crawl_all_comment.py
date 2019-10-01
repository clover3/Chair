import pickle
import os
from path import data_path
from crawl.crawl_uk import get_comment


scope_dir = os.path.join(data_path, "guardian")

def load_article_list():
    root = os.path.join(scope_dir, "by_time")
    out_path = os.path.join(root, "list.pickle")
    return pickle.load(open(out_path, "rb"))


def save(save_path, data):
    pickle.dump(data, open(save_path, "wb"))


def main():
    a_list = load_article_list()
    save_root = os.path.join(scope_dir, "all_comment")
    l_idx_path = os.path.join(save_root, "last_idx")
    start =int(open(l_idx_path, "r").readlines()[-1])
    print(start)
    log_f = open(l_idx_path, "w")

    task = a_list[start:]
    for idx, a in enumerate(task):
        id, short_id = a
        name = short_id.replace("/", "_")
        save_path = os.path.join(save_root, name)
        if not os.path.exists(save_path):
            data = get_comment(short_id)
            save(save_path, data)
        g_idx = idx + start
        if idx%100 == 0:
            print(g_idx)
            log_f.write("{}\n".format(g_idx))
            log_f.flush()

if __name__ == "__main__":
    main()