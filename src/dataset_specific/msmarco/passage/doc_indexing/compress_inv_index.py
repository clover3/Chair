from cache import load_from_pickle
from dataset_specific.msmarco.passage.doc_indexing.build_inverted_index_msmarco import IntInvIndex
from dataset_specific.msmarco.passage.doc_indexing.inv_index_serializer import save_inv_index_as_jsonl, \
    save_inv_index_as_binary, load_inv_index_from_binary, load_inv_index_from_jsonl, save_inv_index_as_binary2, \
    load_inv_index_from_binary2
from trainer_v2.chair_logging import c_log
from cpath import output_path
from misc_lib import path_join


def apply_int_doc_id(inv_index) -> IntInvIndex:
    new_inv_index = {}
    for term, postings in inv_index.items():
        new_postings = []
        for doc_id, cnt in postings:
            new_postings.append((int(doc_id), cnt))

        new_inv_index[term] = new_postings
    return new_inv_index


def small_test():
    inv_index = {'term': [('doc', 1), ('doc', 1), ('doc', 1)], }
    c_log.info("Saving to bytes")
    save_path = "inv_index.bin"
    save_inv_index_as_binary(inv_index, save_path)
    loaded_inv_index = load_inv_index_from_binary(save_path)
    print(loaded_inv_index)


def main():
    c_log.info("Loading pickle")
    inv_index = load_from_pickle("mmp_inv_index_lower")
    c_log.info("Saving to bin using 2")
    save_path = path_join(output_path, "msmarco", "index", "mmp_inv_index_lower.bin")
    save_inv_index_as_binary2(inv_index, save_path)
    c_log.info("Done")


def compress_as_int():
    c_log.info("Loading from binary")
    load_path = path_join(output_path, "msmarco", "index", "mmp_inv_index_lower.bin")
    inv_index = load_inv_index_from_binary(load_path)
    c_log.info("Applying int conversion")
    new_inv_index = apply_int_doc_id(inv_index)
    c_log.info("Saving...")
    save_path = path_join(output_path, "msmarco", "index", "mmp_inv_index_lower_int.bin")
    save_inv_index_as_binary(new_inv_index, save_path)
    c_log.info("Done...")



def main():
    c_log.info("Loading from binary2")
    save_path = path_join(output_path, "msmarco", "index", "mmp_inv_index_lower.bin")
    inv_index = load_inv_index_from_binary2(save_path)
    c_log.info("Done")




if __name__ == "__main__":
    main()
