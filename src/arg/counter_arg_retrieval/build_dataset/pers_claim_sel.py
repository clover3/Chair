from typing import List, Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_claim_perspective_pair, load_perspectrum_golds, PerspectiveCluster
from cpath import at_data_dir


def main():
    pc_data: List[Dict] = load_claim_perspective_pair()

    pc_data.sort(key=lambda e: len(e['perspectives']))
    gold_d: Dict[int, List[PerspectiveCluster]] = load_perspectrum_golds()

    # out_f = open(at_data_dir("perspective", "claims_and_perspective.txt"), "w", encoding="utf-8")
    out_f = open(at_data_dir("perspective", "claims_and_perspective_brief.txt"), "w", encoding="utf-8")
    for e in pc_data:
        cid = e['cId']

        if not gold_d[cid] or len(gold_d[cid]) < 10:
            continue
        text = e['text']
        rows = []
        row = [str(cid), text]
        rows.append(row)

        for pc in gold_d[cid]:
            pid = pc.perspective_ids[0]
            row = [pc.stance_label_5, perspective_getter(pid)]
            rows.append(row)

        for row in rows:
            out_f.write("\t".join(row) + "\n")
        out_f.write("\n")


if __name__ == "__main__":
    main()
