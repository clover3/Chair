from typing import List, Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_claim_perspective_pair, load_perspectrum_golds, PerspectiveCluster
from cpath import at_data_dir


def main():
    pc_data: List[Dict] = load_claim_perspective_pair()

    pc_data.sort(key=lambda e: len(e['perspectives']), reverse=True)
    gold_d: Dict[int, List[PerspectiveCluster]] = load_perspectrum_golds()

    out_f = open(at_data_dir("perspective", "claims_and_perspective.txt"), "w", encoding="utf-8")

    for e in pc_data:
        cid = e['cId']

        if not gold_d[cid]:
            continue
        text = e['text']
        rows = []
        row = [str(cid), text]
        rows.append(row)

        for pc in gold_d[cid]:
            rows.append([pc.stance_label_3, pc.stance_label_5])
            for pid in pc.perspective_ids:
                row = [perspective_getter(pid)]
                rows.append(row)
            rows.append([])

        for row in rows:
            out_f.write("\t".join(row) + "\n")
        out_f.write("\n\n\n")


if __name__ == "__main__":
    main()
