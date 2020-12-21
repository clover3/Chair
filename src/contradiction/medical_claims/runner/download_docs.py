import os
import time

from contradiction.medical_claims.load_corpus import load_all_pmids
from contradiction.medical_claims.load_pubmed_doc import doc_save_dir
from contradiction.medical_claims.nlm_api import e_fetch


def save(pmid, data):
    save_path = os.path.join(doc_save_dir, "{}.xml".format(pmid))
    open(save_path, "wb").write(data)


def main():
    pmid_list = load_all_pmids()
    pmid_list.sort()
    last_success = 0

    request_cnt = 0
    request_time_start = time.time()
    for idx, pmid in enumerate(pmid_list):
        if idx < last_success:
            continue
        try:
            content: bytes = e_fetch(pmid)
            save(pmid, content)
            last_success = idx
        except Exception:
            print("Last success", last_success)
            raise

        request_cnt += 1
        elapsed_from_sleep = time.time() - request_time_start
        if request_cnt >= 10 and elapsed_from_sleep < 1:
            time_to_sleep = 1 - elapsed_from_sleep + 0.1
            time.sleep(time_to_sleep)
            request_cnt = 0
            request_time_start = time.time()


if __name__ == "__main__":
    main()
