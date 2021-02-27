from cpath import at_output_dir


def read_doc_id_title_text():
    doc_id_title_text = at_output_dir("clueweb", "doc_id_title_text.txt")

    all_doc_id = []
    out_d = {}
    for line in open(doc_id_title_text, "r"):
        first_sep = line.find("[SEP]")
        second_sep = line.find("[SEP]", first_sep+1)

        sep_len = len("[SEP]")
        doc_id = line[:first_sep]
        title = line[first_sep+sep_len:second_sep]
        content = line[second_sep+sep_len:]
        all_doc_id.append(doc_id)
        out_d[doc_id] = title, content
    print("num unique docs:", len(set(all_doc_id)))
    return out_d


def main():
    pass



if __name__ == "__main__":
    main()