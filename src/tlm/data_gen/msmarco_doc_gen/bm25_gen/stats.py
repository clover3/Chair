from cpath import at_data_dir
from misc_lib import average
from tlm.data_gen.msmarco_doc_gen.max_sent_encode import regroup_sent_list
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10docMulti


def measure_msmarco_passage():
    msmarco_passage_corpus_path = at_data_dir("msmarco", "collection.tsv")
    passage_dict = {}
    l_list = []
    with open(msmarco_passage_corpus_path, 'r', encoding='utf8') as f:
        for line in f:
            passage_id, text = line.split("\t")
            tokens = text.split()
            l_list.append(len(tokens))

            if len(l_list) > 10000:
                break

    print(average(l_list))


def measure_msmarco_doc():
    split = "train"
    resource = ProcessedResource10docMulti(split)
    job_id = 0
    qids = resource.query_group[job_id]
    l_list = []
    for qid in qids:
        if qid not in resource.get_doc_for_query_d():
            continue

        tokens_d = resource.get_stemmed_tokens_d(qid)
        for doc_id, content in tokens_d.items():
            title, lines = content
            lines = regroup_sent_list(lines, 4)
            for line in lines:
                l_list.append(len(line))
        if len(l_list) > 10000:
            break
    print(average(l_list))


def main():
    print("passage")
    measure_msmarco_passage()
    print("doc")
    measure_msmarco_doc()


if __name__ == "__main__":
    main()