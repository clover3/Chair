from datetime import timedelta
from typing import List

from contradiction.medical_claims.annotation_1.analyze_result.read_batch import load_file_list
from contradiction.medical_claims.annotation_1.analyze_result.view_results import read_hits_empty_scheme
from contradiction.medical_claims.annotation_1.worker_id_info import trusted_worker
from mturk.parse_util import HitResult, parse_mturk_time


def under_time_analysis(hits):
    n_hit = len(hits)
    for t in [30, 60, 120, 180, 300]:
        n_hit_undertime = len(list(filter(lambda h: int(h.work_time) < t, hits)))
        print("{0}\t{1:.2f}".format(t, n_hit_undertime / n_hit))


def chunk_analysis(hits):
    chunk_block_threshold = 60 * 5
    last_submit_time = None
    working_chunks = []
    current_chunk_hits = []
    for h in hits:
        accept_time = parse_mturk_time(h.accept_time)
        submit_time = parse_mturk_time(h.submit_time)
        if last_submit_time is not None:
            since_last_submit = accept_time - last_submit_time
            if since_last_submit > timedelta(
                    seconds=chunk_block_threshold,
            ):
                if current_chunk_hits:
                    duratation = (current_chunk_hits[-1].get_submit_time() - current_chunk_hits[0].get_accept_time())
                    n_hit = len(current_chunk_hits)
                    print("{} hits in {}".format(n_hit, duratation))
                    working_chunks.append(current_chunk_hits)
                    current_chunk_hits = []

                print("")

        else:
            since_last_submit = None
        last_submit_time = submit_time
        current_chunk_hits.append(h)
        duration = submit_time - accept_time
        duration_s = "{0:.0f}".format(duration.seconds/ 60 )
        print("{} / {} / {}".format(accept_time, duration_s, since_last_submit))


def main():
    files = load_file_list()
    all_hits = read_hits_empty_scheme(files)

    target_worker = trusted_worker[0]
    for target_worker in trusted_worker:
        hits: List[HitResult] = list(filter(lambda h: h.worker_id == target_worker, all_hits))
        hits.sort(key=lambda h: h.get_accept_time())
        print(target_worker, len(hits))

        # chunk_analysis(hits)
        # under_time_analysis(hits)


if __name__ == "__main__":
    main()
