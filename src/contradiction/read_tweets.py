import os


def read_en_text(dir_path):
    def readlines(column):
        return open(os.path.join(dir_path, column), "r").readlines()

    lang_list = readlines("lang")
    text_list = readlines("text")

    out_text = []
    for i, text in enumerate(text_list):
        if lang_list[i].strip() == "\"en\"":
            out_text.append(text)

    return out_text

def filter_en():
    dir_path = "/mnt/scratch/youngwookim/data/tweets/2019-01-01"

    out_text = read_en_text(dir_path)
    open(os.path.join(dir_path, "en_text"), "w").writelines(out_text)



if __name__ == '__main__':
    filter_en()