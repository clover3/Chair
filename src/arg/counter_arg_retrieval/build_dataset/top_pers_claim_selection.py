import json
import random
from typing import List, Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_claim_perspective_pair, load_perspectrum_golds, PerspectiveCluster
from cpath import at_data_dir, at_output_dir


def main():
    pc_data: List[Dict] = load_claim_perspective_pair()

    pc_data.sort(key=lambda e: len(e['perspectives']), reverse=True)
    gold_d: Dict[int, List[PerspectiveCluster]] = load_perspectrum_golds()
    ca_cid = 1

    out_j = []
    for e in pc_data[:100]:
        cid = e['cId']
        if not gold_d[cid]:
            continue
        c_text = e['text']
        for pc in gold_d[cid]:
            if random.random() < 0.3:
                first_pid = pc.perspective_ids[0]
                p_text = perspective_getter(first_pid)
                j_entry = {
                    'cid': cid,
                    'claim_text': c_text,
                    'ca_cid': ca_cid,
                    'perspective':
                    {
                        'stance': pc.stance_label_3,
                        'pid': first_pid,
                        'p_text': p_text
                    }
                }
                ca_cid += 1
                out_j.append(j_entry)
    print("total of {}".format(len(out_j)))
    out_f = open(at_output_dir("ca_building", "claims.step1.txt"), "w", encoding="utf-8")
    json.dump(out_j, out_f, indent=True)


if __name__ == "__main__":
    main()
