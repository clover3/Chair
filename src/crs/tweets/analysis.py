from misc_lib import *
import os

def load_all(dir_path):
    data = {}
    for file_path in get_dir_files(dir_path):
        file_name = os.path.split(file_path)[-1]
        lines = open(file_path, encoding='utf-8', errors='ignore').readlines()
        data[file_name] = lines

    return data


def near_duplicate_deletion(texts):
    # if 7 tokens overlap, it is duplicate
    window = 9

    def valid_tokens(tokens):
        output = []
        for t in tokens:
            if t[0] == "@":
                continue
            elif t[:4] == "http":
                continue
            else:
                output.append(t)
        return output




    hash_bin = set()
    hash_owner = dict()

    outputs = []
    for t in texts:
        tokens = valid_tokens(t.split())
        i = 0

        unique = True
        while i + window <= len(tokens) and unique:
            targets = tokens[i:i+window]
            sig = " ".join(targets)
            h = hash(sig)
            if h in hash_bin:
                #print("-----")
                #print(hash_owner[h])
                #print(t)
                unique = False
            else:
                hash_bin.add(h)
                hash_owner[h] = t
            i+= 1
        if unique:
            outputs.append(t)
    return outputs




def work():
    data = load_all("C:\work\Data\controversy_tweets\census")
    all_texts= flatten(data.values())
    print("all text:", len(all_texts))
    uniq_texts = set(all_texts)
    print("unique text:", len(uniq_texts))
    uniq_texts = near_duplicate_deletion(uniq_texts)
    print("unique text:", len(uniq_texts))
    for t in uniq_texts:
        print(t.strip())

if __name__ == '__main__':
    work()


