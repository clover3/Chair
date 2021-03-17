import os
from typing import List, Dict, Tuple

from misc_lib import group_by, TimeEstimator


def get_url_dict_path(disk_id):
    root_dir = "/mnt/nfs/collections/ClueWeb12/ClueWeb12-DocID-To-URL"
    file_name = "ClueWeb12_Disk{}_DocID_To_URL.txt".format(disk_id)
    return os.path.join(root_dir, file_name)


def find_line_start_with(f, keyword, start_loc) -> Tuple[str, int]:
    def readline_after_loc(loc):
        if loc > 0:
            f.seek(loc-1)
            _ = f.readline()
        else:
            f.seek(loc)
        line = f.readline()
        return line

    loc = start_loc
    step = 0
    min_step = 10
    n_iter = 0

    loc_list = []
    def get_next_step_size(last_step_size, direction):
        assert direction in [-1, +1]
        abs_step_size = abs(last_step_size)

        if direction * last_step_size > 0:
            # if same direction
            abs_new_step_size = int(abs_step_size * 1.5)
        elif direction * last_step_size < 0:
            # if opposite direction
            abs_new_step_size = max(int(abs_step_size / 4), min_step)
        elif last_step_size == 0:
            abs_new_step_size = min_step
        else:
            assert False
        return direction * abs_new_step_size

    while True:
        try:
            line = readline_after_loc(loc)
            if not line.strip():
                raise IndexError()
            if line.startswith(keyword):
                return line, loc
            elif line < keyword:
                next_direction = +1
            elif keyword < line :
                next_direction = -1
            else:
                assert False
        except OSError and IndexError:
            if loc > 0:
                next_direction = -1
            else:
                raise
        step = get_next_step_size(step, next_direction)
        next_loc = loc + step
        loc_list.append(loc)
        loc = next_loc
        n_iter += 1

        if n_iter > 1000:
            print("loc", loc)
            print("step", step)
            print("keyword", keyword)
            print(loc_list[-50:])
            raise ValueError("Max iteration reached")


def retrieve_urls(disk_id, doc_ids):
    f = open(get_url_dict_path(disk_id), "r")
    loc = 0
    d = {}
    ticker = TimeEstimator(len(doc_ids))
    for doc_id in doc_ids:
        line, found_loc = find_line_start_with(f, doc_id, loc)
        loc = found_loc
        doc_id_s, url = line.split()
        assert doc_id_s[-1] == ","
        assert doc_id == doc_id_s[:-1]
        d[doc_id] = url
        ticker.tick()
    return d


def get_urls(doc_id_list: List[str]) -> Dict[str, str]:
    doc_id_list.sort()

    def get_disk_id(doc_id):
        st_ed_intervals = [
            (1, "clueweb12-0000tw-00-00000", "clueweb12-0412wb-35-11764"),
            (2, "clueweb12-0500tw-00-00002", "clueweb12-0920wb-00-09365"),
            (3, "clueweb12-1000tw-00-00000", "clueweb12-1416wb-79-16679"),
            (4, "clueweb12-1500tw-00-00000", "clueweb12-1914wb-28-24254")
        ]
        for disk_id, st, ed in st_ed_intervals:
            if st <= doc_id <= ed:
                return disk_id

        raise KeyError(doc_id)

    doc_id_per_disk = group_by(doc_id_list, get_disk_id)

    all_urls = {}
    for disk_id, doc_ids in doc_id_per_disk.items():
        urls = retrieve_urls(disk_id, doc_ids)
        all_urls.update(urls)
    return all_urls


def main():
    doc_id_list = ["clueweb12-0110wb-21-05068",
"clueweb12-0300tw-56-10598",
"clueweb12-0107wb-01-27648",
"clueweb12-0402wb-52-09160",]
    out_d = get_urls(doc_id_list)
    print(out_d)


if __name__ == "__main__":
    main()

