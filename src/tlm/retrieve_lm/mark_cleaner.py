import os
import time

import cpath


def clean_mark():
    for i in range(1000):
        mark_path = os.path.join(cpath.data_path, "adhoc", "seg_rank0_mark", "{}.mark".format(i))
        output_path = os.path.join(cpath.data_path, "stream_pickled", "CandiSet_{}_0".format(i))

        if os.path.exists(mark_path):
            if not os.path.exists(output_path):
                mark_time = os.path.getmtime(mark_path)
                elapsed = time.time() - mark_time

                min_elp = elapsed / 60

                if min_elp > 2 * 60:
                    print("Delete {}.mark : {} min old".format(i, min_elp))
                    os.remove(mark_path)




if __name__ == "__main__":
    while True:
        clean_mark()
        time.sleep(60*60)
