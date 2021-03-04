import json

from tlm.doc_length.sumarize_doc_length import summarize_doc_length


def main():

    def iter_docs():
        input_path = "/mnt/nfs/collections/msmarco/fulldocuments.jsonl"
        num_doc = 100000
        parse_fail = 0
        for line in open(input_path, "r"):
            try:
                j = json.loads(line)
                text = j["document_text"]
                yield text
            except:
                parse_fail += 1
                ##

    summarize_doc_length(iter_docs())



if __name__ == "__main__":
    main()